"""Quantities used by the heuristic to bound the IoU of the candidate
formulas."""

import numpy as np
import torch


def count_dtype(bitmaps: torch.Tensor) -> torch.dtype:
    """Smallest integer type able to count the elements of the bitmaps."""
    num_elements = bitmaps.numel()
    if num_elements < 2**16:
        return torch.int16
    if num_elements < 2**32:
        return torch.int32
    return torch.int64


def compute_neuron_quantities(bitmaps: torch.Tensor, dataset_quantities: tuple) -> tuple:
    """Computes the quantities of the unit mask.

    Args:
        bitmaps (torch.Tensor): boolean unit mask of shape (num_samples, H*W).
        dataset_quantities (tuple): common, unique, and uncoverable pixels of
            the dataset.

    Returns:
        tuple: unit pixels annotated by more concepts, and pixels annotated by
            more concepts where the unit does not fire.
    """
    common_elements, _, _ = dataset_quantities
    # Pixels where the unit does not fire
    unit_not_firing = ~bitmaps

    unit_common = (bitmaps & common_elements).sum(dim=1).cpu().numpy()
    not_firing_common = (unit_not_firing & common_elements).sum(dim=1).cpu().numpy()
    return unit_common, not_firing_common


def compute_concept_quantities(
    concept_mask: torch.Tensor,
    *,
    bitmaps: torch.Tensor,
    dataset_quantities: tuple,
    neuron_quantities: tuple,
) -> dict:
    """Computes the quantities of a concept or formula mask.

    Args:
        concept_mask (torch.Tensor): boolean mask of shape (num_samples, H*W).
        bitmaps (torch.Tensor): boolean unit mask of shape (num_samples, H*W).
        dataset_quantities (tuple): common, unique, and uncoverable bits of the dataset.
        neuron_quantities (tuple): quantities of the unit mask.

    Returns:
        dict: bounds of the quantities of the concept in each sample. The keys
            max_x and min_x are the upper and lower bounds of the quantity x.
            common_intersection and unique_intersection count the pixels
            inside the concept and the unit mask that are annotated by more
            concepts or by one concept. common_extras and unique_extras count
            the same pixels outside the unit mask. common_uncovered counts the
            pixels inside the unit mask and outside the concept that are
            annotated by more concepts. The keys x_sum are the sums of the
            quantities over the samples.
    """
    common_elements, unique_elements, _ = dataset_quantities
    unit_common, _ = neuron_quantities
    dtype = count_dtype(bitmaps)
    concept_mask = concept_mask.to(bitmaps.device)

    intersection = concept_mask & bitmaps
    extras = concept_mask & ~bitmaps
    # Counts per sample
    common_intersection = (intersection & common_elements).sum(dim=1, dtype=dtype).cpu().numpy()
    unique_intersection = (intersection & unique_elements).sum(dim=1, dtype=dtype).cpu().numpy()
    common_extras = (extras & common_elements).sum(dim=1, dtype=dtype).cpu().numpy()
    unique_extras = (extras & unique_elements).sum(dim=1, dtype=dtype).cpu().numpy()
    common_uncovered = unit_common - common_intersection

    # The quantities of a concept are exact, so their max and min coincide
    return {
        "max_common_intersection": common_intersection,
        "min_common_intersection": common_intersection,
        "max_unique_intersection": unique_intersection,
        "min_unique_intersection": unique_intersection,
        "max_common_extras": common_extras,
        "min_common_extras": common_extras,
        "max_unique_extras": unique_extras,
        "min_unique_extras": unique_extras,
        "max_common_uncovered": common_uncovered,
        "min_common_uncovered": common_uncovered,
        "max_common_intersection_sum": common_intersection.sum(),
        "min_common_intersection_sum": common_intersection.sum(),
        "max_unique_intersection_sum": unique_intersection.sum(),
        "min_unique_intersection_sum": unique_intersection.sum(),
        "max_common_extras_sum": common_extras.sum(),
        "min_common_extras_sum": common_extras.sum(),
        "max_unique_extras_sum": unique_extras.sum(),
        "min_unique_extras_sum": unique_extras.sum(),
    }


def compute_iou_from_quantities(concept_quantities: dict, *, num_hits: int) -> float:
    """Returns the IoU of a concept, computed from its quantities.

    Args:
        concept_quantities (dict): quantities of the concept.
        num_hits (int): number of pixels of the unit mask.

    Returns:
        float: the IoU, or 0 if both masks are empty.
    """
    intersection = int(concept_quantities["max_common_intersection_sum"]) + int(
        concept_quantities["max_unique_intersection_sum"]
    )
    union = (
        num_hits
        + int(concept_quantities["max_common_extras_sum"])
        + int(concept_quantities["max_unique_extras_sum"])
    )
    if union == 0:
        return 0.0
    return intersection / union


def compute_heuristic_info(
    masks: list, bitmaps: torch.Tensor, *, dataset_quantities: tuple, disjoint_info: np.ndarray
) -> dict:
    """Computes the information of a unit used by the heuristic.

    Args:
        masks (list): sparse concept masks of shape (num_samples, H*W).
        bitmaps (torch.Tensor): boolean unit mask of shape (num_samples, H*W).
        dataset_quantities (tuple): common, unique, and uncoverable pixels of the dataset.
        disjoint_info (np.ndarray): disjoint matrix of the concepts.

    Returns:
        dict: dataset_quantities, neuron_quantities, concepts_quantities, and
            disjoint_info. The beam searches extend concepts_quantities with
            the quantities of the beam formulas.
    """
    # Move the dataset quantities to the device of the bitmaps once
    common_elements, unique_elements, uncoverable_elements = dataset_quantities
    device_quantities = (
        common_elements.to(bitmaps.device),
        unique_elements.to(bitmaps.device),
        uncoverable_elements.to(bitmaps.device),
    )
    neuron_quantities = compute_neuron_quantities(bitmaps, device_quantities)

    concepts_quantities = []
    for mask in masks:
        concept_mask = torch.from_numpy(mask.toarray())
        concept_quantities = compute_concept_quantities(
            concept_mask,
            bitmaps=bitmaps,
            dataset_quantities=device_quantities,
            neuron_quantities=neuron_quantities,
        )
        concepts_quantities.append(concept_quantities)
    return {
        "dataset_quantities": dataset_quantities,
        "neuron_quantities": neuron_quantities,
        "concepts_quantities": concepts_quantities,
        "disjoint_info": disjoint_info,
    }
