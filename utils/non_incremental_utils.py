"""Utilities of the non-incremental beam search."""

import torch

from compositional import formula as F
from compositional import metrics
from utils import mask_utils


# ============================================================================
# TERMS OF A FORMULA
# ============================================================================

def get_or_terms(formula: F.F) -> list:
    """Returns the terms of the OR chain of the formula. For example,
    (A OR (B AND C)) OR D gives A, B AND C, and D."""
    if isinstance(formula, F.Or):
        return get_or_terms(formula.left) + get_or_terms(formula.right)
    return [formula]


def get_and_terms(formula: F.F) -> list:
    """Returns the terms of the AND chain of the formula. An AND whose right
    side is a NOT is not split. For example, (A AND NOT B) AND C gives
    A AND NOT B, and C."""
    if isinstance(formula, F.And) and not isinstance(formula.right, F.Not):
        return get_and_terms(formula.left) + get_and_terms(formula.right)
    return [formula]


# ============================================================================
# COMBINATIONS OF TWO FORMULAS
# ============================================================================

def combine_formulas(formula_a: F.F, formula_b: F.F) -> list:
    """Returns the formulas obtained by combining two formulas.

    Args:
        formula_a (F.F): first formula.
        formula_b (F.F): second formula.

    Returns:
        list: the combined formulas.
    """
    combined_formulas = []
    combined_formulas.extend(combine_same_operator(formula_a, formula_b))
    combined_formulas.extend(combine_disjoint(formula_a, formula_b))
    combined_formulas.extend(combine_different_operators(formula_a, formula_b))
    return combined_formulas


def chain_terms(terms: list, operator: type) -> F.F:
    """Joins the terms with the operator from left to right."""
    chained_formula = terms[0]
    for term in terms[1:]:
        chained_formula = operator(chained_formula, term)
    return chained_formula


def combine_disjoint(formula_a: F.F, formula_b: F.F) -> list:
    """Returns the AND and the OR of two formulas without shared leaves."""
    shared_leaves = set(formula_a.get_leaves()) & set(formula_b.get_leaves())
    if len(shared_leaves) > 0:
        return []
    return [F.And(formula_a, formula_b), F.Or(formula_a, formula_b)]


def apply_distributivity(formula_a: F.F, formula_b: F.F, operator: type) -> F.F:
    """Applies the distributive property to two formulas that use only the
    operator. For example, (A AND B) and (A AND C) give A AND (B OR C), and
    (A OR B) and (A OR C) give A OR (B AND C).
    """
    shared_leaves = set(formula_a.get_leaves()) & set(formula_b.get_leaves())
    opposite_operator = F.Or if operator == F.And else F.And

    all_leaves = sorted(set(formula_a.get_leaves() + formula_b.get_leaves()))
    shared_terms = [leaf for leaf in all_leaves if leaf in shared_leaves]
    other_terms = [leaf for leaf in all_leaves if leaf not in shared_leaves]

    factored_formula = chain_terms(shared_terms, operator)
    remaining_formula = chain_terms(other_terms, opposite_operator)
    return operator(factored_formula, remaining_formula)


def combine_same_operator(formula_a: F.F, formula_b: F.F) -> list:
    """Combines two formulas that share a leaf and use the same operator. For
    example, (A OR B) and (A OR C) give A OR B OR C and A OR (B AND C).
    """
    # Distinct operators, starting from the one of the root
    ops_a = list(dict.fromkeys(formula_a.get_ops()))
    ops_b = list(dict.fromkeys(formula_b.get_ops()))
    if len(ops_a) != len(ops_b):
        return []
    both_full_not = F.is_full_not(formula_a) and F.is_full_not(formula_b)
    if not both_full_not and len(ops_a) != 1:
        return []
    # The operator of the root must be the same. It is AND for a chain of AND NOT
    if ops_a[0] != ops_b[0]:
        return []
    operator = F.And if ops_a[0] == "AND" else F.Or

    shared_leaves = set(formula_a.get_leaves()) & set(formula_b.get_leaves())
    if len(shared_leaves) == 0:
        # Formulas without shared leaves are combined by combine_disjoint
        return []

    # Sorting puts the negated concepts last, so NOT is always on the right of AND
    all_leaves = sorted(set(formula_a.get_leaves() + formula_b.get_leaves()))
    chained_formula = chain_terms(all_leaves, operator)
    if "NOT" in ops_a or "NOT" in ops_b:
        return [chained_formula]
    distributed_formula = apply_distributivity(formula_a, formula_b, operator)
    return [chained_formula, distributed_formula]


def combine_different_operators(formula_a: F.F, formula_b: F.F) -> list:
    """Combines a formula that uses only OR and a formula that uses only AND
    when they share exactly one leaf. The shared leaf of the OR formula is
    replaced by the AND formula. For example, (A OR B) and (A AND C) give
    (A AND C) OR B.
    """
    ops_a = list(dict.fromkeys(formula_a.get_ops()))
    ops_b = list(dict.fromkeys(formula_b.get_ops()))
    if len(ops_a) != 1 or len(ops_b) != 1:
        return []
    if ops_a[0] == ops_b[0]:
        return []

    shared_leaves = set(formula_a.get_leaves()) & set(formula_b.get_leaves())
    if len(shared_leaves) != 1:
        # Formulas without shared leaves are combined by combine_disjoint.
        # With more shared leaves, the formulas are in contradiction.
        return []

    if ops_a[0] == "OR":
        or_formula, and_formula = formula_a, formula_b
    else:
        or_formula, and_formula = formula_b, formula_a

    combined_formula = and_formula
    for leaf in or_formula.get_leaves():
        if leaf not in shared_leaves:
            combined_formula = F.Or(combined_formula, leaf)
    return [combined_formula]


# ============================================================================
# HISTORY OF THE SEARCH
# ============================================================================

def combine_beam_formulas(beam_formulas: list) -> list:
    """Returns the combinations of every pair of different formulas of the beam.

    Args:
        beam_formulas (list): formulas of the beam.

    Returns:
        list: the combined formulas.
    """
    combinations = []
    for i in range(len(beam_formulas) - 1):
        for j in range(i + 1, len(beam_formulas)):
            first_formula = beam_formulas[i]
            second_formula = beam_formulas[j]
            if first_formula == second_formula:
                continue
            new_formulas = combine_formulas(first_formula, second_formula)
            # Remove duplicates
            combinations.extend(list(set(new_formulas)))
    return combinations


def combine_with_history(beam_formulas: list, history: dict) -> list:
    """Returns the combinations of each formula of the beam with the formulas
    of the history that are shorter than it. The atomic concepts of the history
    are excluded.

    Args:
        beam_formulas (list): formulas of the beam.
        history (dict): formulas of the history by length.

    Returns:
        list: the combined formulas.
    """
    combinations = []
    for beam_formula in beam_formulas:
        for history_length in range(2, len(beam_formula)):
            for history_formula in history.get(history_length, []):
                if beam_formula == history_formula:
                    continue
                new_formulas = combine_formulas(beam_formula, history_formula)
                # Remove duplicates
                combinations.extend(list(set(new_formulas)))
    return combinations


def update_history(history: dict, new_formulas: list, *, current_length: int, max_length: int):
    """Adds the new formulas to the history.

    Args:
        history (dict): formulas of the history by length, updated in place.
        new_formulas (list): formulas to add.
        current_length (int): length of the current step. Only longer formulas
            are added.
        max_length (int): maximum length of the explanations.
    """
    for formula in new_formulas:
        formula_length = len(formula)
        if current_length < formula_length <= max_length:
            history.setdefault(formula_length, []).append(formula)


# ============================================================================
# FUNCTIONAL EQUIVALENCE
# ============================================================================

def find_equivalent_formulas(formula_mask: torch.Tensor, formula_iou: float, beam_info: dict) -> list:
    """Returns the formulas of the beam that are functionally equivalent to a
    formula.

    Args:
        formula_mask (torch.Tensor): mask of the formula.
        formula_iou (float): IoU of the formula. Only the formulas with the
            same IoU are compared.
        beam_info (dict): mask and IoU of each formula of the beam.

    Returns:
        list: the equivalent formulas of the beam.
    """
    equivalent_formulas = []
    for beam_formula, (beam_mask, beam_iou) in beam_info.items():
        if beam_iou != formula_iou:
            continue
        if metrics.iou(beam_mask, formula_mask) == 1.0:
            equivalent_formulas.append(beam_formula)
    return equivalent_formulas


def have_same_leaves(formula_a: F.F, formula_b: F.F) -> bool:
    """Returns True if the two formulas use the same leaves."""
    leaves_a = set(formula_a.get_leaves())
    leaves_b = set(formula_b.get_leaves())
    return len(leaves_a) > 0 and leaves_a == leaves_b


def is_verified(formula: F.F, masks: list, *, device: torch.device) -> bool:
    """Returns True if the data supports every AND NOT of the formula. For
    example, (A OR B) AND NOT C is supported if C overlaps both A and B.

    Args:
        formula (F.F): the formula.
        masks (list): sparse concept masks of shape (num_samples, H*W).
        device (torch.device): device where the masks are computed.
    """
    for term in get_or_terms(formula):
        if not (isinstance(term, F.And) and isinstance(term.right, F.Not)):
            continue
        negated_mask = mask_utils.get_formula_mask(term.right.val, masks, device=device)
        for left_term in get_or_terms(term.left):
            left_mask = mask_utils.get_formula_mask(left_term, masks, device=device)
            if metrics.iou(left_mask, negated_mask) == 0.0:
                return False
    return True
