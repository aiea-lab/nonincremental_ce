"""Wrappers of the probed models."""

import os
from typing import Callable

import torchvision
import torch
import numpy as np

from utils import dataset_utils
from src import nlp_models, settings, snli

class ModelWrapper:
    """Wrapper of a probed model."""

    def __init__(self):
        self.model = None
        self.data_loader = None
        self.iter_data = None

    def set_loader(self, cfg: settings.Settings):
        """Sets the data loader of the dataset."""
        raise NotImplementedError

    # Adapted from loadmodel of CoEx (vision/loader/model_loader.py)
    def hook(self, hook_fn: Callable, feature_names: list) -> list:
        """Registers hook_fn as a forward hook of the given modules.

        Args:
            hook_fn (callable): hook function.
            feature_names (list): names of the modules.

        Returns:
            list: handles of the hooks.
        """
        handles = []
        for name in feature_names:
            if isinstance(name, list):
                # Iteratively retrieve the module
                hook_model = self.model
                for n in name:
                    hook_model = hook_model._modules.get(n)
            else:
                hook_model = self.model._modules.get(name)
            if hook_model is None:
                raise ValueError(f"Couldn't find feature {name}")
            handles.append(hook_model.register_forward_hook(hook_fn))
        return handles

    def get_layer_activations(self, *, layer: str, layer_file: str) -> torch.Tensor:
        """Checks if the activations are already computed and saved, otherwise
        computes them and saves them.

        Args:
            layer (str): name of the layer.
            layer_file (str): path of the activations.

        Returns:
            torch.Tensor: activations of the layer.
        """
        if self.data_loader is None:
            raise ValueError("Data loader not set. Please call set_loader before loading the activations.")
        total_activations = []
        if os.path.exists(layer_file):
            total_activations = np.load(layer_file)
        else:
            print(f"Computing activations for layer {layer}")
            activations = self.compute_activations(
               layer
            )
            activations = activations[0].numpy()
            print(f"Saving activations for layer {layer} to {layer_file}")
            with open(layer_file, "wb") as f:
                np.save(f, activations)
            total_activations = activations
        total_activations = torch.from_numpy(total_activations)
        return total_activations


class Place365Model(ModelWrapper):
    """Wrapper of a torchvision model trained on Places365."""

    def __init__(self, *, model_name: str, weights: str, device: str):
        """
        Args:
            model_name (str): name of the model.
            weights (str): path of the checkpoint.
            device (str): device of the model.
        """
        if "densenet" in model_name:
            raise NameError("Please use DenseNetPlace365 class for DenseNet models")
        super().__init__()
        self.input_size = 227 if "alexnet" in model_name else 224
        self.model = self.load_checkpoint(model_name=model_name, weights=weights, device=device)

    def set_loader(self, cfg: settings.Settings):
        """Sets the data loader of the images.

        Args:
            cfg (settings.Settings): settings of the run.
        """
        self.data_loader = dataset_utils.get_data_loader(cfg)
        self.iter_data = dataset_utils.get_data_iter_fn(cfg.get_dataset_name())
        
    def load_checkpoint(self, *, model_name: str, weights: str, device: str) -> torch.nn.Module:
        """
        Loads the model from the checkpoint.

        Args:
            model_name (str): name of the model.
            weights (str): path of the checkpoint.
            device (str): device of the model.

        Returns:
            torch.nn.Module: the model.
        """

        print(f"Loading model:{model_name}\n\tfrom {weights}")

        model_fn = torchvision.models.__dict__[model_name]

        checkpoint = torch.load(weights, map_location=device)
        model = model_fn(num_classes=365)
        # Remove the prefix added by the data parallel layer
        state_dict = {
            str.replace(k, "module.", ""): v
            for k, v in checkpoint["state_dict"].items()
        }

        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        print(f"{model_name} loaded in {device}. Modality: Evaluation")
        return model


   
    @torch.no_grad()
    def compute_activations(self, layer: str) -> torch.Tensor:
        """Returns the activations of the layer for the images of the data
        loader.

        Args:
            layer (str): name of the layer.

        Returns:
            torch.Tensor: activations of the layer.
        """
        device = next(self.model.parameters()).device
        temp_activations = []

        def hook_feature(module: torch.nn.Module, inp: tuple, output: torch.Tensor):
            temp_activations.append(output.data.cpu())

        handles = self.hook(hook_feature, [layer])

        activations = []
        for data in self.data_loader:
            # Transformations
            images = self.iter_data(data)

            if len(images.shape) == 4:
                images = images.squeeze(0)

            if len(images.shape) == 3:
                images = images.unsqueeze(0)

            
            # Move to GPU
            images = images.to(device)

            # Forward pass
            _ = self.model(images)

            # collect data
            activations.append(temp_activations[0].cpu())

            # Empty the temp list
            del temp_activations[:]
            temp_activations = []

        for handle in handles:
            handle.remove()
        activations = torch.unsqueeze(
            torch.cat(activations), dim=0
        )
        return activations


    
class DenseNetPlace365(Place365Model):
    """Wrapper of a DenseNet trained on Places365."""

    def __init__(self, *, model_name: str, weights: str, device: str):
        """
        Args:
            model_name (str): name of the model.
            weights (str): path of the checkpoint.
            device (str): device of the model.
        """
        ModelWrapper.__init__(self)
        self.model = self.load_checkpoint(model_name=model_name, weights=weights, device=device)
        self.input_size = 224

    def load_checkpoint(self, *, model_name: str, weights: str, device: str) -> torch.nn.Module:
        """
        Loads the model from the checkpoint.

        Args:
            model_name (str): name of the model.
            weights (str): path of the checkpoint.
            device (str): device of the model.

        Returns:
            torch.nn.Module: the model.
        """
        def rep(k: str) -> str:
            for i in range(6):
                k = k.replace(f"norm.{i}", f"norm{i}")
                k = k.replace(f"relu.{i}", f"relu{i}")
                k = k.replace(f"conv.{i}", f"conv{i}")
            return k
        print(f"Loading model:{model_name}\n\tfrom {weights}")

        model_fn = torchvision.models.__dict__[model_name]

        checkpoint = torch.load(weights, map_location=device)
        model = model_fn(num_classes=365)
        # Fix the old names of the DenseNet layers
        state_dict = checkpoint.state_dict()
        state_dict = {rep(k): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        
        model = model.to(device)
        model.eval()
        print(f"{model_name} loaded in {device}. Modality: Evaluation")
        return model

class NLPModelWrapper(ModelWrapper):
    """Wrapper of a model trained on SNLI."""

    def __init__(self, *, model_name: str, weights: str, device: str):
        """
        Args:
            model_name (str): name of the model.
            weights (str): path of the checkpoint.
            device (str): device of the model.
        """
        super().__init__()
        self.vocab = None
        self.model = self.load_checkpoint(model_name=model_name, weights=weights, device=device)

    def set_loader(self, cfg: settings.Settings):
        """Sets the data loader of the SNLI analysis dataset.

        Args:
            cfg (settings.Settings): settings of the run.
        """
        self.data_loader = dataset_utils.get_data_loader(cfg, self.vocab)
        self.iter_data = dataset_utils.get_data_iter_fn(cfg.get_dataset_name())

    @torch.no_grad()
    def compute_activations(self, layer: str) -> torch.Tensor:
        """Returns the activations of the layer for the samples of the data
        loader.

        Args:
            layer (str): name of the layer. Only mlp is supported.

        Returns:
            torch.Tensor: activations of shape (1, num_samples, num_units).
        """
        if layer != "mlp":
            raise ValueError(f"Layer {layer} not supported for the NLI models. Accepted values: mlp")
        device = next(self.model.parameters()).device

        activations = []
        for batch in self.data_loader:
            premises, premise_lengths, hypotheses, hypothesis_lengths, _ = self.iter_data(batch)
            premises = premises.to(device)
            hypotheses = hypotheses.to(device)

            states = self.model.get_final_reprs(premises, premise_lengths, hypotheses, hypothesis_lengths)
            activations.append(states.cpu())

        activations = torch.unsqueeze(torch.cat(activations), dim=0)
        return activations


class TransformersWrapper(NLPModelWrapper):
    """Wrapper of a transformer model trained on SNLI, which reads the text of
    the sentences rebuilt from their word ids."""

    def set_loader(self, cfg: settings.Settings):
        """Sets the data loader of the SNLI analysis dataset, tokenizing the
        batches.

        Args:
            cfg (settings.Settings): settings of the run.
        """
        self.data_loader = dataset_utils.get_data_loader(cfg, self.vocab)
        self.iter_data = self.tokenize_batch

    def tokenize_batch(self, batch: tuple) -> tuple:
        """Returns the tokenized premises and hypotheses of a batch of the loader."""
        premises, _, hypotheses, _, sentence_indices = dataset_utils.decompose_batch_nli(batch)
        premise_texts = self.ids_to_texts(premises)
        hypothesis_texts = self.ids_to_texts(hypotheses)
        tokenizer = self.model.tokenizer
        premise_tokens = tokenizer(premise_texts, return_tensors="pt", padding=True, truncation=True)
        hypothesis_tokens = tokenizer(hypothesis_texts, return_tensors="pt", padding=True, truncation=True)
        return premise_tokens, None, hypothesis_tokens, None, sentence_indices

    def ids_to_texts(self, word_ids: torch.Tensor) -> list:
        """Rebuilds the sentences from their word ids."""
        ignored_ids = {self.vocab["stoi"][snli.UNK_TOKEN], snli.PAD_IDX}
        texts = []
        for sentence_ids in word_ids.T.tolist():
            words = [self.vocab["itos"][word_id] for word_id in sentence_ids if word_id not in ignored_ids]
            texts.append(" ".join(words))
        return texts


class Llama(TransformersWrapper):
    """Wrapper of the LLaMA classifier."""

    def load_checkpoint(self, *, model_name: str, weights: str, device: str) -> torch.nn.Module:
        """Loads the LLaMA classifier from the checkpoint."""
        print(f"Loading model:{model_name}")
        checkpoint = torch.load(weights, map_location="cpu")
        self.vocab = {"itos": checkpoint["itos"], "stoi": checkpoint["stoi"]}
        model = nlp_models.LLAMAEntailmentClassifier()
        model.load_state_dict(checkpoint["state_dict"])
        model = model.to(device)
        model.eval()
        print(f"{model_name} loaded in {device}. Modality: Evaluation")
        return model


class BERT(TransformersWrapper):
    """Wrapper of the BERT classifier."""

    def load_checkpoint(self, *, model_name: str, weights: str, device: str) -> torch.nn.Module:
        """Loads the BERT classifier from the checkpoint."""
        print(f"Loading model:{model_name}")
        checkpoint = torch.load(weights, map_location="cpu")
        self.vocab = {"itos": checkpoint["itos"], "stoi": checkpoint["stoi"]}
        model = nlp_models.BertEntailmentClassifier()
        model.load_state_dict(checkpoint["state_dict"])
        model = model.to(device)
        model.eval()
        print(f"{model_name} loaded in {device}. Modality: Evaluation")
        return model


class Bowman(NLPModelWrapper):
    """Wrapper of the BiLSTM classifier."""

    def load_checkpoint(self, *, model_name: str, weights: str, device: str) -> torch.nn.Module:
        """
        Loads the model from the checkpoint.

        Args:
            model_name (str): name of the model.
            weights (str): path of the checkpoint.
            device (str): device of the model.

        Returns:
            torch.nn.Module: the model.
        """

        print(f"Loading model:{model_name}")
        ckpt = torch.load(weights, map_location="cpu")

        # Load vocab from trained model
        self.vocab = {"itos": ckpt["itos"], "stoi": ckpt["stoi"]}
        
        # Load model
        model_enc = nlp_models.TextEncoder(len(ckpt["stoi"]))
        model_clf = nlp_models.BowmanEntailmentClassifier
        model = model_clf(model_enc)
        model.load_state_dict(ckpt["state_dict"])
        model = model.to(device)
        model.eval()
        print(f"{model_name} loaded in {device}. Modality: Evaluation")
        return model
