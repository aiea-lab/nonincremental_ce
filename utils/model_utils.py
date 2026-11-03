"""Utilities to load the probed models and to select the units to explain."""

import random
from typing import Optional

import torch

from src import settings
from src.models import ModelWrapper, Place365Model, DenseNetPlace365, Bowman, BERT, Llama


def load_model(cfg: settings.Settings) -> ModelWrapper:
    """Loads the probed model and sets its data loader.

    Args:
        cfg (settings.Settings): settings of the run.

    Returns:
        src.models.ModelWrapper: the wrapper of the model.
    """
    weights = cfg.get_weights()
    
    if cfg.model == 'densenet161':
        model_wrapper = DenseNetPlace365(model_name=cfg.model, weights=weights, device=cfg.device)
    elif cfg.model == 'alexnet' or cfg.model == 'resnet18':
        model_wrapper = Place365Model(model_name=cfg.model, weights=weights, device=cfg.device)
    elif cfg.model == 'bowman':
        model_wrapper = Bowman(model_name=cfg.model, weights=weights, device=cfg.device)
    elif cfg.model == 'bert':
        model_wrapper = BERT(model_name=cfg.model, weights=weights, device=cfg.device)
    elif cfg.model == 'llama':
        model_wrapper = Llama(model_name=cfg.model, weights=weights, device=cfg.device)
    else:
        raise ValueError("Pretrained model not supported")

    # Set the data loader
    model_wrapper.set_loader(cfg)
    return model_wrapper


def get_selected_units(
    layer_activations: torch.Tensor, *, units: Optional[list] = None, random_units: int = 0, seed: int = 0
) -> list:
    """Returns the units to explain.

    Args:
        layer_activations (torch.Tensor): activations of the layer.
        units (list): units to explain. If None, the units are extracted
            randomly or all the units are explained.
        random_units (int): number of units extracted randomly. If 0, all the
            units are explained.
        seed (int): seed of the random extraction.

    Returns:
        list: indices of the selected units.
    """
    num_units = layer_activations.shape[1]
    if units is not None:
        for unit in units:
            if int(unit) >= num_units:
                raise ValueError(f"Unit {unit} is out of range")
        selected_units = [int(unit) for unit in units]
    elif random_units > 0:
        generator = random.Random(seed)
        selected_units = generator.sample(range(num_units), random_units)
    else:
        selected_units = list(range(num_units))
    return selected_units
