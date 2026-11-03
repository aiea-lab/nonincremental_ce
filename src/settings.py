"""Settings of a run. Adapted from
https://github.com/jayelm/compexp/blob/master/vision/settings.py"""


import os
from typing import Optional


class Settings:
    """
    Class that stores all the settings used in each run.
    """

    def __init__(
        self,
        *,
        model: str,
        layer: str,
        device: str,
        dataset: str = "broden",
        root_models: str = "data/model",
        root_datasets: str = "data/datasets",
        root_segmentations: str = "data/cache/segmentations/",
        root_activations: str = "data/cache/activations/",
        step_size: int = 80,
        fast_impl: bool = False,
        root_results: str = "data/results",
    ):
        """
        Args:
            model (str): name of the probed model.
            layer (str): layer of the model whose units are explained.
            device (str): device used for the computations.
            dataset (str): name of the dataset, broden or snli.
            root_models (str): directory of the model checkpoints.
            root_datasets (str): directory of the datasets.
            root_segmentations (str): directory of the cached concept masks.
            root_activations (str): directory of the cached activations.
            step_size (int): number of concepts whose masks are extracted in a
                single step.
            fast_impl (bool): whether to load all the dataset segmentations in
                memory once to extract the concept masks.
            root_results (str): directory of the explanations.
        """
        self.set_dataset_name(dataset)
        self.set_model_name(model)
        self.set_layer(layer)
        self.set_model_root(root_models)
        self.set_dataset_root(root_datasets)
        self.set_activations_root(root_activations)
        self.set_segmentations_root(root_segmentations)
        self.__root_results = root_results
        self.__step_size = step_size
        self.__fast_impl = fast_impl
        self.set_device(device)

    def set_dataset_name(self, dataset_name: str):
        """
        Sets the name of the dataset.
        """
        self.dataset = dataset_name

    def set_model_name(self, model_name: str):
        """
        Sets the name of the model.
        """
        self.model = model_name

    def set_layer(self, layer: str):
        """
        Sets the layer to analyze.
        """
        self.layer = layer

    def set_model_root(self, model_root: str):
        """
        Sets the root directory where the models are stored.
        """
        self.__root_models = model_root
        if not os.path.exists(self.__root_models):
            os.makedirs(self.__root_models)

    def set_dataset_root(self, dataset_root: str):
        """
        Sets the root directory where the datasets are stored.
        """
        self.dir_datasets = dataset_root
        if not os.path.exists(self.dir_datasets):
            os.makedirs(self.dir_datasets)

    def set_activations_root(self, activations_root: str):
        """
        Sets the root directory where the activations are stored.
        """
        self.__root_activations = activations_root
        if not os.path.exists(self.__root_activations):
            os.makedirs(self.__root_activations)

    def set_segmentations_root(self, segmentations_root: str):
        """
        Sets the root directory where the concept masks are stored.
        """
        self.__root_segmentations = segmentations_root

    def get_segmentations_dir(self) -> str:
        """
        Returns the directory where the concept masks are stored.
        """
        return os.path.join(
            self.__root_segmentations, self.dataset, str(self.get_annotations_size()), "masks"
        )

    def get_info_dir(self) -> str:
        """
        Returns the directory where the information about the concept masks is
        stored.
        """
        if self.get_dataset_name() == "snli":
            return os.path.join(self.__root_segmentations, self.dataset, "info")
        return os.path.join(
            self.__root_segmentations, self.dataset, str(self.get_annotations_size()), "info"
        )

    def get_results_dir(self) -> str:
        """
        Returns the directory where the explanations of the layer are stored.
        """
        return os.path.join(self.__root_results, self.dataset, self.model, self.layer)

    def get_step_size(self) -> int:
        """
        Returns the number of concepts whose masks are extracted in a single step.
        """
        return self.__step_size

    def fast_impl_is_enabled(self) -> bool:
        """
        Returns whether the fast mask extraction is enabled.
        """
        return self.__fast_impl

    def set_device(self, device: str):
        """
        Sets the device to use for the model.
        """
        self.device = device

    def get_dataset_name(self) -> str:
        """
        Returns the name of the dataset.
        """
        return self.dataset

    def get_device(self) -> str:
        """
        Returns the device to use for the model.
        """
        return self.device

    def get_dataset_root(self) -> str:
        """
        Returns the root directory where the datasets are stored.
        """
        return self.dir_datasets

    def get_activations_root(self) -> str:
        """
        Returns the root directory where the activations are stored.
        """
        return self.__root_activations

    def get_img_size(self) -> int:
        """
        Returns the size of the images given to the model.
        """
        if self.model == "resnet18" or self.model == "densenet161":
            return 224
        elif self.model == "alexnet":
            return 227
        else:
            raise ValueError(f"Unknown image size for Model {self.model} on dataset {self.dataset}")

    def get_annotations_size(self) -> int:
        """
        Returns the resolution of the Broden annotations, which is the same
        for every model (224) following the Optimal CoEx paper.
        """
        return 224

    def get_mask_shape(self) -> Optional[tuple]:
        """
        Returns the shape of the masks of a sample, or None if a mask has one
        value per sample.
        """
        if self.get_dataset_name() == "snli":
            return None
        if self.model in ["resnet18", "densenet161", "alexnet"]:
            return (112, 112)
        else:
            raise ValueError(f"Unknown mask shape for Model {self.model} on dataset {self.dataset}")

    def get_image_mean(self) -> list:
        """
        Returns the mean of the dataset.
        """
        if self.model in ["resnet18", "densenet161", "alexnet"]:
            return [0.485, 0.456, 0.406]
        else:
            raise ValueError(f"Unknown image mean for Model {self.model} on dataset {self.dataset}")

    def get_image_stdev(self) -> list:
        """
        Returns the standard deviation of the dataset.
        """
        if self.model in ["resnet18", "densenet161", "alexnet"]:
            return [0.229, 0.224, 0.225]
        else:
            raise ValueError(f"Unknown image stdev for Model {self.model} on dataset {self.dataset}")

    def get_weights(self) -> str:
        """
        Returns the path to the weights of the model.
        """
        if self.get_dataset_name() == "broden":
            if self.model == "densenet161":
                model_file_name = (
                    "whole_densenet161_places365_python36.pth.tar"
                )
                return os.path.join(self.get_model_root(), "zoo", model_file_name)
            elif self.model == "alexnet" or self.model == "resnet18":
                model_file_name = f"{self.model}_places365.pth.tar"
                return os.path.join(self.get_model_root(), "zoo", model_file_name)
            else:
                raise ValueError(f"Model {self.model} not supported for dataset {self.dataset}")
        elif self.get_dataset_name() == "snli":
            nli_model_files = {"bowman": "bowman.pth", "bert": "bert_dense.pth", "llama": "llama_dense.pth"}
            if self.model in nli_model_files:
                return os.path.join(self.get_model_root(), "other", nli_model_files[self.model])
            else:
                raise ValueError(f"Model {self.model} not supported for dataset {self.dataset}")

    def get_model_root(self) -> str:
        """
        Returns the root directory where the models are stored.
        """
        return self.__root_models

    def get_activations_file_path(self) -> str:
        """
        Returns the path to the activations of the layer.
        """
        activation_dir = self.get_activations_root() + f"/{self.dataset}/{self.model}/"
        if not os.path.exists(activation_dir):
            os.makedirs(activation_dir)
        activation_path = os.path.join(activation_dir,  f"{self.layer}.npy")
        return activation_path
