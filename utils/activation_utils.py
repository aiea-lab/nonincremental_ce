"""Utilities to turn the activations of a unit into a binary activation mask."""

from typing import Optional

import torch

from src import constants
from utils import vecquantile


def has_enough_activations(unit_activations: torch.Tensor) -> bool:
    """Returns True if the unit has enough non-zero activations to be explained.

    Args:
        unit_activations (torch.Tensor): activations of the unit, of shape
            (num_samples, H, W) or (num_samples,).
    """
    num_samples = unit_activations.shape[0]
    num_nonzero_activations = torch.count_nonzero(unit_activations).item()
    min_nonzero_activations = num_samples * constants.NETDISSECT_QUANTILE
    return num_nonzero_activations >= min_nonzero_activations


def compute_threshold(unit_activations: torch.Tensor, *, batch_size: int = 64, seed: int = 1) -> float:
    """Returns the activation threshold of the unit. Only the top
    constants.NETDISSECT_QUANTILE fraction of the activations of the unit is
    above it.

    Args:
        unit_activations (torch.Tensor): activations of the unit, of shape
            (num_samples, H, W) or (num_samples,).
        batch_size (int): number of samples added to the estimator at a time.
        seed (int): seed of the quantile estimator.

    Returns:
        float: the threshold.
    """
    quantile_vector = vecquantile.QuantileVector(depth=1, seed=seed)
    for start in range(0, unit_activations.shape[0], batch_size):
        batch = unit_activations[start : start + batch_size]
        quantile_vector.add(batch.flatten().reshape(-1, 1).numpy())
    quantiles = quantile_vector.readout(1000)[0]
    return float(quantiles[int(1000 * (1 - constants.NETDISSECT_QUANTILE) - 1)])


def compute_bitmaps(unit_activations: torch.Tensor, threshold: float, mask_shape: Optional[tuple]) -> torch.Tensor:
    """Returns the activation mask of the unit.

    Args:
        unit_activations (torch.Tensor): activations of the unit, of shape
            (num_samples, H, W) or (num_samples,).
        threshold (float): activation threshold of the unit.
        mask_shape (tuple): shape (H, W) of the concept masks, or None if a
            mask has one value per sample.

    Returns:
        torch.Tensor: boolean mask of shape (num_samples, H*W), or
            (num_samples, 1) if mask_shape is None.
    """
    if mask_shape is None:
        bitmaps = unit_activations > threshold
        return bitmaps.unsqueeze(1)

    upsampled_activations = torch.nn.functional.interpolate(
        unit_activations.unsqueeze(1), size=mask_shape, mode="bilinear"
    ).squeeze(1)
    bitmaps = upsampled_activations.double() > threshold
    return bitmaps.reshape(bitmaps.shape[0], -1)
