"""Non-incremental beam search for compositional explanations."""

from collections import Counter
import heapq
from typing import Optional

import torch

from compositional import formula as F
from compositional import heuristic_info, metrics
from utils import mask_utils, non_incremental_utils, search_utils


# ============================================================================
# EXPANSION
# ============================================================================

def add_concept_to_each_term(terms: list, *, concept: F.Leaf, operation: str, chain_operator: type) -> list:
    """Returns the formulas obtained by adding the concept to each term of the
    chain."""
    expanded_formulas = []
    for index, term in enumerate(terms):
        new_terms = terms[:index] + [search_utils.add_concept(term, concept=concept, operation=operation)] + terms[index + 1 :]
        expanded_formulas.append(non_incremental_utils.chain_terms(new_terms, chain_operator))
    return expanded_formulas


def violates_constraints(formula: F.F, constraints: Optional[list]) -> bool:
    """Returns whether the formula contains two concepts that cannot be
    combined."""
    if constraints is None:
        return False
    concepts_in_formula = set(formula.get_vals())
    for concept in concepts_in_formula:
        if any(constrained in concepts_in_formula for constrained in constraints[concept]):
            return True
    return False


def expand_formula_non_incremental(
    formula: F.F, *, candidate_labels: list, non_zero_labels: set, constraints: Optional[list] = None
) -> list:
    """Returns the incremental and non-incremental expansions of the formula.

    Args:
        formula (F.F): formula to expand.
        candidate_labels (list): atomic concepts that can be added.
        non_zero_labels (set): concepts whose IoU with the unit is positive.
        constraints (list): concepts that cannot be combined with each
            concept, or None.
    """
    excluded_concepts = search_utils.get_excluded_concepts(formula, constraints)
    or_terms = non_incremental_utils.get_or_terms(formula)
    and_terms = non_incremental_utils.get_and_terms(formula)

    expanded_formulas = []
    for candidate in candidate_labels:
        if candidate.val in excluded_concepts:
            continue
        for operation in ["OR", "AND", "NOT"]:
            if operation != "NOT" and candidate.val not in non_zero_labels:
                continue
            # Incremental expansion
            expanded_formulas.append(search_utils.add_concept(formula, concept=candidate, operation=operation))

            # Non-incremental expansion
            if operation == "OR" and len(and_terms) > 1:
                expanded_formulas.extend(
                    add_concept_to_each_term(and_terms, concept=candidate, operation=operation, chain_operator=F.And)
                )
            elif operation != "OR" and len(or_terms) > 1:
                expanded_formulas.extend(
                    add_concept_to_each_term(or_terms, concept=candidate, operation=operation, chain_operator=F.Or)
                )
    return expanded_formulas


# ============================================================================
# BEAM SELECTION
# ============================================================================

def manage_equivalence(
    candidate: F.F, candidate_mask: torch.Tensor, candidate_iou: float, *, masks: list, beam_info: dict
) -> tuple:
    """Manages the formulas of the beam that are functionally equivalent to
    the candidate.

    Args:
        candidate (F.F): the candidate formula.
        candidate_mask (torch.Tensor): mask of the candidate.
        candidate_iou (float): IoU of the candidate with the unit.
        masks (list): sparse concept masks of shape (num_samples, H*W).
        beam_info (dict): maps each formula of the beam to (mask, IoU).

    Returns:
        tuple: formulas to add to the beam, formulas to remove from the beam,
            and combinations of the candidate with its equivalent formulas.
    """
    formulas_to_add = {}
    formulas_to_remove = set()
    combinations = []

    equivalent_formulas = non_incremental_utils.find_equivalent_formulas(candidate_mask, candidate_iou, beam_info)
    if len(equivalent_formulas) == 0:
        formulas_to_add[candidate] = (candidate_iou, candidate_mask)
        return formulas_to_add, formulas_to_remove, combinations

    device = candidate_mask.device
    for equivalent_formula in equivalent_formulas:
        if non_incremental_utils.have_same_leaves(candidate, equivalent_formula):
            # Same concepts. Keep the formula verified by the data
            if non_incremental_utils.is_verified(equivalent_formula, masks, device=device):
                continue
            if non_incremental_utils.is_verified(candidate, masks, device=device):
                formulas_to_remove.add(equivalent_formula)
                formulas_to_add[candidate] = (candidate_iou, candidate_mask)
        else:
            if len(equivalent_formula) < len(candidate) and equivalent_formula == candidate.left:
                # The candidate does not change the mask of the formula
                continue
            # Different concepts. Keep both and combine them
            formulas_to_add[candidate] = (candidate_iou, candidate_mask)
            combinations.extend(non_incremental_utils.combine_formulas(candidate, equivalent_formula))
    return formulas_to_add, formulas_to_remove, combinations


def select_beam_functional_aware(
    frontier: list,
    *,
    masks: list,
    beam_masks: dict,
    bitmaps: torch.Tensor,
    beam_limit: int,
    previous_beam: dict,
) -> tuple:
    """Selects the next beam, managing the functionally equivalent formulas.

    Args:
        frontier (list): heap of candidates (-estimated IoU, formula).
        masks (list): sparse concept masks of shape (num_samples, H*W).
        beam_masks (dict): masks already computed for some formulas.
        bitmaps (torch.Tensor): boolean unit mask of shape (num_samples, H*W).
        beam_limit (int): maximum number of nodes of the beam.
        previous_beam (dict): nodes (IoU, formula) of the previous beam,
            mapped to their IoU.

    Returns:
        tuple: nodes of the new beam, number of visited candidates, and
            combinations of equivalent formulas.
    """
    # Mask and IoU of each formula of the beam
    beam_info = {}
    for (_, formula), formula_iou in previous_beam.items():
        formula_mask = mask_utils.get_formula_mask(
            formula, masks, beam_masks, device=bitmaps.device
        )
        beam_info[formula] = (formula_mask, formula_iou)

    # Heap of the beam, with the worst node first
    beam = []
    minimum_iou = 0
    for node, node_iou in previous_beam.items():
        if len(beam) < beam_limit:
            heapq.heappush(beam, node)
        elif node_iou > minimum_iou:
            heapq.heapreplace(beam, node)
        minimum_iou = beam[0][0]
    minimum_iou = beam[0][0] if beam else 0

    visited = 0
    combinations = []
    # Avoid visiting the same formula twice
    recent_formulas = set()
    recent_estimation = -1
    while len(frontier) > 0:
        negated_estimated_iou, formula = heapq.heappop(frontier)
        estimated_iou = -negated_estimated_iou
        # Same stop rule of the incremental selection. Candidates with the same
        # IoU as the worst formula are not checked for functional equivalence
        if len(beam) >= beam_limit and estimated_iou <= minimum_iou:
            break

        if estimated_iou == recent_estimation:
            if formula in recent_formulas:
                continue
            recent_formulas.add(formula)
        else:
            recent_formulas = {formula}
            recent_estimation = estimated_iou

        formula_mask = mask_utils.get_formula_mask(
            formula, masks, beam_masks, device=bitmaps.device
        )
        formula_iou = metrics.iou(formula_mask, bitmaps)
        visited += 1
        if len(beam) >= beam_limit and formula_iou <= minimum_iou:
            continue

        formulas_to_add, formulas_to_remove, new_combinations = (
            manage_equivalence(
                formula, formula_mask, formula_iou, masks=masks, beam_info=beam_info
            )
        )
        combinations.extend(new_combinations)
        for formula_to_remove in formulas_to_remove:
            # Equivalent formulas have the same IoU as the candidate
            beam.remove((formula_iou, formula_to_remove))
            del beam_info[formula_to_remove]
            heapq.heapify(beam)
        for formula_to_add, (iou_to_add, mask_to_add) in formulas_to_add.items():
            heapq.heappush(beam, (iou_to_add, formula_to_add))
            beam_info[formula_to_add] = (mask_to_add, iou_to_add)

        # Remove the worst nodes until the beam is within its limit
        while len(beam) > beam_limit:
            _, removed_formula = heapq.heappop(beam)
            del beam_info[removed_formula]
        minimum_iou = beam[0][0]
    return list(beam), visited, combinations


# ============================================================================
# SEARCH
# ============================================================================

def non_incremental_beam_search(
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
    use_non_incremental_expansion: bool = True,
    use_recombinations: bool = True,
    use_functional_equivalence: bool = True,
) -> tuple:
    """Computes the explanation of a unit with the non-incremental beam
    search.

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
        use_non_incremental_expansion (bool): whether to use the
            non-incremental expansion.
        use_recombinations (bool): whether to combine the formulas.
        use_functional_equivalence (bool): whether to manage the functionally
            equivalent formulas.

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

    # Initialize the history with the best concepts and their combinations
    history = {}
    if use_recombinations:
        history[1] = [label for label, label_iou in concepts_iou.most_common(beam_size) if label_iou > 0]
        first_combinations = non_incremental_utils.combine_beam_formulas(history[1])
        non_incremental_utils.update_history(history, first_combinations, current_length=1, max_length=length)

    if use_non_incremental_expansion:
        expand_formula = expand_formula_non_incremental
    else:
        expand_formula = search_utils.expand_formula

    beam_masks = {}
    total_visited = 0
    expanded_nodes = 0
    estimated_nodes = 0
    for current_length in range(1, length):
        next_length = current_length + 1

        # Expand the formulas of the beam added at the previous step
        candidates = []
        for _, formula in beam:
            if len(formula) == current_length:
                candidates.extend(
                    expand_formula(
                        formula,
                        candidate_labels=candidate_labels,
                        non_zero_labels=non_zero_labels,
                        constraints=constraints,
                    )
                )
                expanded_nodes += 1
        # Add the formulas of the history with the next length that satisfy the constraints
        for history_formula in history.get(next_length, []):
            if not violates_constraints(history_formula, constraints):
                candidates.append(history_formula)
        estimated_nodes += len(candidates)

        # Visit the candidates in order of estimated IoU and update the beam
        frontier = search_utils.estimate_frontier(candidates, estimator=estimator)
        if use_functional_equivalence:
            next_beam, visited_nodes, equivalent_combinations = select_beam_functional_aware(
                frontier,
                masks=masks,
                beam_masks=beam_masks,
                bitmaps=bitmaps,
                beam_limit=beam_size,
                previous_beam=beam,
            )
        else:
            next_beam, visited_nodes = search_utils.select_beam(
                frontier,
                masks=masks,
                beam_masks=beam_masks,
                bitmaps=bitmaps,
                beam_limit=beam_size,
                previous_beam=beam,
            )
            equivalent_combinations = []
        total_visited += visited_nodes
        if use_recombinations:
            # Combinations of equivalent formulas with different concepts
            non_incremental_utils.update_history(
                history, equivalent_combinations, current_length=next_length, max_length=length
            )

        for node_iou, formula in next_beam:
            if len(formula) == next_length:
                beam[(node_iou, formula)] = node_iou
        beam = dict(Counter(beam).most_common(beam_size))

        # Update the history with the new formulas of the beam and their combinations
        if use_recombinations:
            history[next_length] = [formula for _, formula in next_beam if len(formula) == next_length]
            if next_length < length:
                beam_combinations = non_incremental_utils.combine_beam_formulas(history[next_length])
                history_combinations = non_incremental_utils.combine_with_history(history[next_length], history)
                non_incremental_utils.update_history(
                    history,
                    beam_combinations + history_combinations,
                    current_length=next_length,
                    max_length=length,
                )

        # Quantities of the new beam formulas, used by the heuristic
        beam_formulas = [formula for _, formula in beam]
        estimator.add_beam_formulas(beam_formulas, beam_masks=beam_masks)

    (_, best_formula), best_iou = Counter(beam).most_common(1)[0]
    return best_formula, best_iou, total_visited, expanded_nodes, estimated_nodes
