"""Utilities for the beam searches.
"""

import heapq
import queue
from typing import Optional

import torch

from compositional import estimators, metrics
from compositional import formula as F
from utils import mask_utils


def compute_iou(formula: F.F, masks: list, bitmaps: torch.Tensor, beam_masks: Optional[dict] = None) -> float:
    """Returns the exact IoU between the formula mask and the unit mask."""
    formula_mask = mask_utils.get_formula_mask(formula, masks, beam_masks, device=bitmaps.device)
    return metrics.iou(formula_mask, bitmaps)


def get_excluded_concepts(formula: F.F, constraints: Optional[list]) -> set:
    """Returns the concepts that cannot be added to the formula.

    Args:
        formula (F.F): formula to expand.
        constraints (list): concepts that cannot be combined with each
            concept, or None.
    """
    concepts_in_formula = set(formula.get_vals())
    excluded_concepts = set(concepts_in_formula)
    if constraints is not None:
        for concept in concepts_in_formula:
            excluded_concepts.update(constraints[concept])
    return excluded_concepts


def add_concept(formula: F.F, *, concept: F.Leaf, operation: str) -> F.F:
    """Returns formula OR concept, formula AND concept, or formula AND NOT concept."""
    if operation == "OR":
        return F.Or(formula, concept)
    if operation == "AND":
        return F.And(formula, concept)
    if operation == "NOT":
        return F.And(formula, F.Not(concept))
    raise ValueError(f"Unknown operation {operation}")


def expand_formula(
    formula: F.F, *, candidate_labels: list, non_zero_labels: set, constraints: Optional[list] = None
) -> list:
    """Returns the incremental expansions of the formula.

    Args:
        formula (F.F): formula to expand.
        candidate_labels (list): atomic concepts that can be added.
        non_zero_labels (set): concepts whose IoU with the unit is positive.
        constraints (list): concepts that cannot be combined with each
            concept, or None.
    """
    excluded_concepts = get_excluded_concepts(formula, constraints)
    expanded_formulas = []
    for candidate in candidate_labels:
        if candidate.val in excluded_concepts:
            continue
        for operation in ["OR", "AND", "NOT"]:
            if operation != "NOT" and candidate.val not in non_zero_labels:
                continue
            expanded_formulas.append(add_concept(formula, concept=candidate, operation=operation))
    return expanded_formulas


def estimate_frontier(candidates: list, *, estimator: estimators.Estimator) -> list:
    """Returns the candidates with a positive estimated IoU as a heap.

    Args:
        candidates (list): candidate formulas.
        estimator (estimators.Estimator): estimator of the IoU of the candidates.

    Returns:
        list: heap of the candidates and their negated estimated IoU.
    """
    frontier = []
    for candidate in candidates:
        max_iou = estimator.estimate_max_iou(candidate)
        if max_iou > 0:
            frontier.append((-max_iou, candidate))
    heapq.heapify(frontier)
    return frontier


def select_beam(
    frontier: list,
    *,
    masks: list,
    beam_masks: dict,
    bitmaps: torch.Tensor,
    beam_limit: int,
    previous_beam: dict,
) -> tuple:
    """Selects the next beam among the previous beam and the candidates of
    the frontier.

    Returns:
        tuple: nodes of the new beam and number of visited candidates.
    """
    # Priority queue of the beam, with the worst node first
    beam = queue.PriorityQueue(beam_limit)
    for node, node_iou in previous_beam.items():
        if not beam.full():
            beam.put(node)
        elif node_iou > beam.queue[0][0]:
            beam.get()
            beam.put(node)
    minimum_iou = 0 if beam.empty() else beam.queue[0][0]

    visited = 0
    while len(frontier) > 0:
        node = heapq.heappop(frontier)
        negated_estimated_iou, formula = node
        if beam.full() and -negated_estimated_iou <= minimum_iou:
            break
        formula_iou = compute_iou(formula, masks, bitmaps, beam_masks)
        visited += 1
        node = (formula_iou, formula)
        if not beam.full():
            beam.put(node)
            minimum_iou = beam.queue[0][0]
        elif formula_iou > minimum_iou:
            beam.get()
            beam.put(node)
            minimum_iou = beam.queue[0][0]
    return list(beam.queue), visited
