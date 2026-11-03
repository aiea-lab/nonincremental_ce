""" MMESH heuristic. 
"""

import os

import torch
from tqdm import tqdm

from compositional import formula as F
from src import constants


# ============================================================================
# GEOMETRY OF THE MASKS
# ============================================================================

def get_bounding_boxes(masks: torch.Tensor) -> torch.Tensor:
    """Returns the bounding box of the mask of each sample.

    Args:
        masks (torch.Tensor): boolean masks of shape (num_samples, H, W).

    Returns:
        torch.Tensor: rectangles of shape (num_samples, 2, 2). Empty masks get
            an empty rectangle.
    """
    num_samples, num_rows, num_columns = masks.shape
    rows_with_pixels = masks.any(dim=2)
    columns_with_pixels = masks.any(dim=1)
    row_indices = torch.arange(num_rows, device=masks.device)
    column_indices = torch.arange(num_columns, device=masks.device)

    # First and last row and column with a pixel
    top = torch.where(rows_with_pixels, row_indices, num_rows).min(dim=1).values
    bottom = torch.where(rows_with_pixels, row_indices, -1).max(dim=1).values
    left = torch.where(columns_with_pixels, column_indices, num_columns).min(dim=1).values
    right = torch.where(columns_with_pixels, column_indices, -1).max(dim=1).values
    # Empty masks get an empty rectangle
    is_empty = bottom < 0
    top = torch.where(is_empty, 0, top)
    left = torch.where(is_empty, 0, left)
    return torch.stack([torch.stack([top, left], dim=1), torch.stack([bottom, right], dim=1)], dim=1)


def get_inscribed_rectangles(masks: torch.Tensor) -> torch.Tensor:
    """Returns the largest rectangle inscribed in the mask of each sample.

    Args:
        masks (torch.Tensor): boolean masks of shape (num_samples, H, W).

    Returns:
        torch.Tensor: rectangles of shape (num_samples, 2, 2). Empty masks get
            an empty rectangle.
    """
    num_samples, num_rows, num_columns = masks.shape
    device = masks.device

    # Columns where the horizontal run of each pixel starts and ends. The end
    # is excluded.
    run_start = torch.zeros(num_samples, num_rows, num_columns, dtype=torch.long, device=device)
    for column in range(1, num_columns):
        run_start[:, :, column] = torch.where(
            masks[:, :, column - 1], run_start[:, :, column - 1], column
        )
    run_end = torch.full((num_samples, num_rows, num_columns), num_columns, dtype=torch.long, device=device)
    for column in range(num_columns - 2, -1, -1):
        run_end[:, :, column] = torch.where(
            masks[:, :, column + 1], run_end[:, :, column + 1], column + 1
        )

    # Candidate rectangle of each column, updated row by row
    height = torch.zeros(num_samples, num_columns, dtype=torch.long, device=device)
    left = torch.zeros(num_samples, num_columns, dtype=torch.long, device=device)
    right = torch.full((num_samples, num_columns), num_columns, dtype=torch.long, device=device)

    best_area = torch.zeros(num_samples, dtype=torch.long, device=device)
    best_rectangles = torch.tensor([[0, 0], [-1, -1]], device=device).repeat(num_samples, 1, 1)
    for row in range(num_rows):
        row_mask = masks[:, row, :]
        height = torch.where(row_mask, height + 1, 0)
        left = torch.where(row_mask, torch.maximum(left, run_start[:, row, :]), 0)
        right = torch.where(row_mask, torch.minimum(right, run_end[:, row, :]), num_columns)
        areas = height * (right - left)

        # Largest candidate of the row
        best_column = areas.argmax(dim=1, keepdim=True)
        row_best_area = areas.gather(1, best_column).squeeze(1)
        improved = row_best_area > best_area
        best_height = height.gather(1, best_column).squeeze(1)
        best_left = left.gather(1, best_column).squeeze(1)
        best_right = right.gather(1, best_column).squeeze(1)
        row_rectangles = torch.stack(
            [
                torch.stack([row - best_height + 1, best_left], dim=1),
                torch.stack([torch.full_like(best_left, row), best_right - 1], dim=1),
            ],
            dim=1,
        )
        best_rectangles = torch.where(improved[:, None, None], row_rectangles, best_rectangles)
        best_area = torch.where(improved, row_best_area, best_area)
    return best_rectangles


def get_rectangles_overlap(rectangles_a: torch.Tensor, rectangles_b: torch.Tensor) -> torch.Tensor:
    """Returns the overlap area of two rectangles for each sample.

    Args:
        rectangles_a (torch.Tensor): rectangles of shape (num_samples, 2, 2).
        rectangles_b (torch.Tensor): rectangles of shape (num_samples, 2, 2).

    Returns:
        torch.Tensor: overlap areas of shape (num_samples,).
    """
    top = torch.maximum(rectangles_a[:, 0, 0], rectangles_b[:, 0, 0])
    left = torch.maximum(rectangles_a[:, 0, 1], rectangles_b[:, 0, 1])
    bottom = torch.minimum(rectangles_a[:, 1, 0], rectangles_b[:, 1, 0])
    right = torch.minimum(rectangles_a[:, 1, 1], rectangles_b[:, 1, 1])
    overlap_height = torch.clamp(bottom - top + 1, min=0)
    overlap_width = torch.clamp(right - left + 1, min=0)
    return overlap_height * overlap_width


# ============================================================================
# INFORMATION ABOUT THE MASKS
# ============================================================================

def compute_masks_info(masks: torch.Tensor, *, mask_shape: tuple) -> dict:
    """Returns the area, the largest inscribed rectangle, and the bounding box
    of the mask of each sample.

    Args:
        masks (torch.Tensor): boolean masks of shape (num_samples, H*W).
        mask_shape (tuple): shape (H, W) of the mask of a sample.

    Returns:
        dict: "areas" of shape (num_samples,), "inscribed" and "boxes" of shape
            (num_samples, 2, 2).
    """
    num_samples = masks.shape[0]
    areas = masks.sum(dim=1, dtype=torch.int32)
    empty_rectangle = torch.tensor([[0, 0], [-1, -1]], dtype=torch.int32, device=masks.device)
    inscribed = empty_rectangle.repeat(num_samples, 1, 1)
    boxes = empty_rectangle.repeat(num_samples, 1, 1)

    # Rectangles of the non-empty masks, computed in batches
    non_empty_samples = torch.nonzero(areas > 0).squeeze(1)
    for start in range(0, len(non_empty_samples), constants.MMESH_BATCH_SIZE):
        samples = non_empty_samples[start:start + constants.MMESH_BATCH_SIZE]
        samples_masks = masks[samples].reshape(-1, mask_shape[0], mask_shape[1])
        inscribed[samples] = get_inscribed_rectangles(samples_masks).to(torch.int32)
        boxes[samples] = get_bounding_boxes(samples_masks).to(torch.int32)
    return {"areas": areas, "inscribed": inscribed, "boxes": boxes}


def get_concepts_info(masks: list, *, mask_shape: tuple, info_dir: str, device: str) -> dict:
    """Computes or loads the MMESH information of the concepts.

    Args:
        masks (list): sparse concept masks of shape (num_samples, H*W).
        mask_shape (tuple): shape (H, W) of the mask of a sample.
        info_dir (str): directory where the information is cached.
        device (str): device where the information is computed.

    Returns:
        dict: "areas" of shape (num_concepts, num_samples), "inscribed" and
            "boxes" of shape (num_concepts, num_samples, 2, 2).
    """
    file_path = os.path.join(info_dir, "mmesh_info.pt")
    if os.path.exists(file_path):
        return torch.load(file_path)

    num_concepts = len(masks)
    num_samples = masks[0].shape[0]
    concepts_info = {
        "areas": torch.zeros(num_concepts, num_samples, dtype=torch.int32),
        "inscribed": torch.zeros(num_concepts, num_samples, 2, 2, dtype=torch.int32),
        "boxes": torch.zeros(num_concepts, num_samples, 2, 2, dtype=torch.int32),
    }
    for concept, mask in enumerate(tqdm(masks, desc="Computing the MMESH information")):
        concept_mask = torch.from_numpy(mask.toarray()).to(device)
        concept_info = compute_masks_info(concept_mask, mask_shape=mask_shape)
        for key, value in concept_info.items():
            concepts_info[key][concept] = value.cpu()

    os.makedirs(info_dir, exist_ok=True)
    torch.save(concepts_info, file_path)
    return concepts_info


# ============================================================================
# MMESH ESTIMATE
# ============================================================================

def get_term_info(term: F.F, unit_info: dict) -> dict:
    """Returns the MMESH information of a concept or of a beam formula."""
    if not isinstance(term, F.Leaf):
        return unit_info["formulas"][term]
    concept = term.val
    concepts_info = unit_info["concepts"]
    return {
        "areas": concepts_info["areas"][concept],
        "inscribed": concepts_info["inscribed"][concept],
        "boxes": concepts_info["boxes"][concept],
        "intersections": unit_info["concepts_intersections"][concept],
    }


def estimate_max_iou(formula: F.F, *, unit_info: dict, num_hits: int) -> float:
    """Returns the MMESH estimate of the IoU of a candidate formula.

    Args:
        formula (F.F): the candidate formula.
        unit_info (dict): MMESH information of the unit.
        num_hits (int): number of pixels of the unit mask.

    Returns:
        float: the estimated IoU.
    """
    neuron_areas = unit_info["neuron_areas"]
    max_area = unit_info["max_area"]
    is_and_not = isinstance(formula, F.And) and isinstance(formula.right, F.Not)
    right_term = formula.right.val if is_and_not else formula.right
    left_info = get_term_info(formula.left, unit_info)
    right_info = get_term_info(right_term, unit_info)

    left_areas = left_info["areas"]
    left_intersections = left_info["intersections"]
    if is_and_not:
        # Area and intersection of NOT right
        right_areas = max_area - right_info["areas"]
        right_intersections = neuron_areas - right_info["intersections"]
    else:
        right_areas = right_info["areas"]
        right_intersections = right_info["intersections"]

    # Samples where the formula can be present
    if isinstance(formula, F.Or):
        formula_is_in = (left_areas > 0) | (right_areas > 0)
    else:
        formula_is_in = (left_areas > 0) & (right_areas > 0)
    left_intersections = left_intersections * formula_is_in
    right_intersections = right_intersections * formula_is_in

    # A mask behaves like a scene if, in each sample, it is either empty or full
    left_is_scene = bool(((left_areas == 0) | (left_areas == max_area)).all())
    right_is_scene = bool(((right_areas == 0) | (right_areas == max_area)).all())

    # Upper bound of the intersection and lower bound of the area of the formula
    if left_is_scene or right_is_scene:
        # Exact values if one of the two masks is a scene
        if isinstance(formula, F.Or):
            formula_areas = torch.maximum(left_areas, right_areas)
            intersections = torch.where(
                formula_areas == max_area, neuron_areas, left_intersections + right_intersections
            )
        else:
            formula_areas = torch.minimum(left_areas, right_areas)
            intersections = torch.minimum(left_intersections, right_intersections)
    elif isinstance(formula, F.Or):
        intersections = torch.minimum(neuron_areas, left_intersections + right_intersections)
        boxes_overlap = get_rectangles_overlap(left_info["boxes"], right_info["boxes"])
        formula_areas = torch.maximum(left_areas, right_areas)
        formula_areas = torch.maximum(formula_areas, left_areas + right_areas - boxes_overlap)
        formula_areas = torch.maximum(formula_areas, intersections)
    elif is_and_not:
        intersections = torch.minimum(left_intersections, right_intersections)
        boxes_overlap = get_rectangles_overlap(left_info["boxes"], right_info["boxes"])
        formula_areas = torch.maximum(left_areas - boxes_overlap, intersections)
    else:
        intersections = torch.minimum(left_intersections, right_intersections)
        inscribed_overlap = get_rectangles_overlap(left_info["inscribed"], right_info["inscribed"])
        formula_areas = torch.maximum(inscribed_overlap, intersections)

    intersection = int(intersections.sum())
    union = num_hits + int(formula_areas.sum()) - intersection
    if union == 0:
        return 0.0
    return intersection / union
