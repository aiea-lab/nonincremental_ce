"""Metrics to evaluate the alignment between formulas and units."""

import torch


def iou(mask_a: torch.Tensor, mask_b: torch.Tensor) -> float:
    """Returns the intersection over union of two boolean masks."""
    intersection = torch.count_nonzero(mask_a & mask_b).item()
    union = torch.count_nonzero(mask_a | mask_b).item()
    if union == 0:
        # Both masks are empty
        return 0.0
    return intersection / union


def detection_accuracy(*, formula_mask: torch.Tensor, unit_mask: torch.Tensor) -> float:
    """Returns the detection accuracy, which is the fraction of the formula mask
    covered by the unit mask."""
    intersection = torch.count_nonzero(formula_mask & unit_mask).item()
    formula_size = torch.count_nonzero(formula_mask).item()
    return intersection / formula_size


def activation_coverage(*, formula_mask: torch.Tensor, unit_mask: torch.Tensor) -> float:
    """Returns the activation coverage, which is the fraction of the unit mask
    covered by the formula mask."""
    intersection = torch.count_nonzero(formula_mask & unit_mask).item()
    unit_size = torch.count_nonzero(unit_mask).item()
    return intersection / unit_size
