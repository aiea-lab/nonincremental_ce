"""Incremental beam search for compositional explanations."""

from collections import Counter
from typing import Optional

import torch

from compositional import formula as F
from compositional import heuristic_info
from utils import search_utils


def incremental_beam_search(
    masks: list,
    bitmaps: torch.Tensor,
    unit_heuristic_info: dict,
    *,
    beam_size: int,
    length: int,
    first_beam_size: int,
    estimator_class: type,
    estimator_args: dict,
    constraints: Optional[list] = None,
) -> tuple:
    """Computes the explanation of a unit with the incremental beam search.

    Args:
        masks (list): sparse concept masks of shape (num_samples, H*W).
        bitmaps (torch.Tensor): boolean unit mask of shape (num_samples, H*W).
        unit_heuristic_info (dict): information of the
            unit used by the heuristic.
        beam_size (int): number of formulas kept in the beam.
        length (int): maximum length of the formulas.
        first_beam_size (int): number of concepts in the first beam.
        estimator_class (type): subclass of estimators.Estimator used to
            estimate the IoU of the candidates.
        estimator_args (dict): arguments of the estimator besides the ones
            of the unit.
        constraints (list): concepts that cannot be combined with each
            concept, or None.

    Returns:
        tuple: best formula, its IoU, and the number of visited, expanded, and
            estimated nodes.
    """
    num_hits = int(bitmaps.sum().item())
    candidate_labels = [F.Leaf(concept) for concept in range(len(masks))]
    estimator = estimator_class(
        masks=masks,
        bitmaps=bitmaps,
        unit_heuristic_info=unit_heuristic_info,
        max_length=length,
        **estimator_args,
    )

    # The first beam contains the concepts with the highest IoU
    concepts_quantities = unit_heuristic_info["concepts_quantities"]
    concepts_iou = Counter()
    for label in candidate_labels:
        concepts_iou[label] = heuristic_info.compute_iou_from_quantities(
            concepts_quantities[label.val], num_hits=num_hits
        )
    non_zero_labels = {label.val for label, label_iou in concepts_iou.items() if label_iou > 0}
    first_beam_num = min(len(concepts_iou), first_beam_size)
    beam = {
        (label_iou, label): label_iou
        for label, label_iou in concepts_iou.most_common(first_beam_num)
        if label_iou > 0
    }

    beam_masks = {}
    total_visited = 0
    expanded_nodes = 0
    estimated_nodes = 0
    for current_length in range(1, length):
        # Expand the formulas of the beam added at the previous step
        candidates = []
        for _, formula in beam:
            if len(formula) == current_length:
                candidates.extend(
                    search_utils.expand_formula(
                        formula,
                        candidate_labels=candidate_labels,
                        non_zero_labels=non_zero_labels,
                        constraints=constraints,
                    )
                )
                expanded_nodes += 1
        # Remove the duplicates
        candidates = list(set(candidates))
        estimated_nodes += len(candidates)

        # Visit the candidates in order of estimated IoU and update the beam
        frontier = search_utils.estimate_frontier(candidates, estimator=estimator)
        next_beam, visited_nodes = search_utils.select_beam(
            frontier,
            masks=masks,
            beam_masks=beam_masks,
            bitmaps=bitmaps,
            beam_limit=beam_size,
            previous_beam=beam,
        )
        total_visited += visited_nodes
        for node_iou, formula in next_beam:
            if len(formula) == current_length + 1:
                beam[(node_iou, formula)] = node_iou
        beam = dict(Counter(beam).most_common(beam_size))

        # Information of the new beam formulas, used by the estimator
        beam_formulas = [formula for _, formula in beam]
        estimator.add_beam_formulas(beam_formulas, beam_masks=beam_masks)

    (_, best_formula), best_iou = Counter(beam).most_common(1)[0]
    return best_formula, best_iou, total_visited, expanded_nodes, estimated_nodes
