"""Utilities to compute the mask of a formula and the disjoint concepts."""

from typing import Optional

import numpy as np
import scipy.sparse as sparse
import torch

from compositional import formula as F


def get_formula_mask(
    formula: F.F, masks: list, beam_masks: Optional[dict] = None, device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    """Returns the mask of a formula, combining the masks of its concepts.

    Args:
        formula (F.F): the formula.
        masks (list): sparse concept masks of shape (num_samples, H*W).
        beam_masks (dict): masks already computed for some formulas.
        device (torch.device): device where the mask is computed.

    Returns:
        torch.Tensor: boolean mask of shape (num_samples, H*W).
    """
    if beam_masks is not None and formula in beam_masks:
        return beam_masks[formula].to(device)
    if isinstance(formula, F.Leaf):
        return torch.from_numpy(masks[formula.val].toarray()).to(device)
    if isinstance(formula, F.Or):
        left_mask = get_formula_mask(formula.left, masks, beam_masks, device)
        right_mask = get_formula_mask(formula.right, masks, beam_masks, device)
        return left_mask | right_mask
    if isinstance(formula, F.And):
        left_mask = get_formula_mask(formula.left, masks, beam_masks, device)
        right_mask = get_formula_mask(formula.right, masks, beam_masks, device)
        return left_mask & right_mask
    if isinstance(formula, F.Not):
        return ~get_formula_mask(formula.val, masks, beam_masks, device)
    raise ValueError(f"Unknown formula type {type(formula)}")


def disjoint_matrix_from_masks(masks: list) -> np.ndarray:
    """Returns the matrix of the disjoint concepts. Two concepts are disjoint
    if their masks never share a pixel.

    Args:
        masks (list): sparse concept masks of shape (num_samples, H*W).

    Returns:
        np.ndarray: boolean matrix whose entry (i, j) is True if the concepts
            i and j are disjoint.
    """
    stacked = sparse.vstack([mask.reshape(1, -1) for mask in masks]).tocsr()
    overlaps = (stacked @ stacked.T).toarray()
    disjoint_matrix = ~overlaps.astype(bool)
    np.fill_diagonal(disjoint_matrix, False)
    return disjoint_matrix
