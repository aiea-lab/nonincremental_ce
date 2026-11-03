"""Heuristic that bounds the IoU of the candidate formulas."""

from typing import Optional

import numpy as np

from compositional import formula as F


def combine_quantities(
    formula: F.F, *, left_quantities: dict, right_quantities: dict, neuron_quantities: tuple
) -> Optional[dict]:
    """Estimates the quantities of the formula from the ones of its operands,
    when the operands may overlap. For AND NOT, right_quantities are the ones
    of the negated concept.

    Returns:
        dict: estimated quantities, or None if the formula cannot
            improve on its operands.
    """
    unit_common, not_firing_common = neuron_quantities
    left, right = left_quantities, right_quantities

    if isinstance(formula, F.Or):
        max_unique_intersection = left["max_unique_intersection"] + right["max_unique_intersection"]
        min_unique_intersection = left["min_unique_intersection"] + right["min_unique_intersection"]
        max_unique_intersection_sum = left["max_unique_intersection_sum"] + right["max_unique_intersection_sum"]
        min_unique_intersection_sum = left["min_unique_intersection_sum"] + right["min_unique_intersection_sum"]
        # Common pixels may be shared by the two operands
        min_common_intersection = np.maximum(left["min_common_intersection"], right["min_common_intersection"])
        max_common_intersection = np.minimum(
            left["max_common_intersection"] + right["max_common_intersection"], unit_common
        )
        # Discard the formula if it cannot improve on one of its operands
        no_improvement_on_left = (
            max_unique_intersection_sum <= left["max_unique_intersection_sum"]
            and min_unique_intersection_sum <= left["min_unique_intersection_sum"]
            and np.all(max_common_intersection <= left["max_common_intersection"])
            and np.all(min_common_intersection <= left["min_common_intersection"])
        )
        no_improvement_on_right = (
            max_unique_intersection_sum <= right["max_unique_intersection_sum"]
            and min_unique_intersection_sum <= right["min_unique_intersection_sum"]
            and np.all(max_common_intersection <= right["max_common_intersection"])
            and np.all(min_common_intersection <= right["min_common_intersection"])
        )
        if no_improvement_on_left or no_improvement_on_right:
            return None
        min_unique_extras = left["min_unique_extras"] + right["min_unique_extras"]
        max_unique_extras = left["max_unique_extras"] + right["max_unique_extras"]
        min_unique_extras_sum = left["min_unique_extras_sum"] + right["min_unique_extras_sum"]
        max_unique_extras_sum = left["max_unique_extras_sum"] + right["max_unique_extras_sum"]
        min_common_extras = np.maximum(left["min_common_extras"], right["min_common_extras"])
        max_common_extras = np.minimum(not_firing_common, left["max_common_extras"] + right["max_common_extras"])

    elif isinstance(formula, F.And) and isinstance(formula.right, F.Not):
        # Removing a concept does not change the unique pixels of the left operand
        max_unique_intersection = left["max_unique_intersection"]
        min_unique_intersection = left["min_unique_intersection"]
        max_unique_intersection_sum = left["max_unique_intersection_sum"]
        min_unique_intersection_sum = left["min_unique_intersection_sum"]
        # Common unit pixels not covered by the negated concept
        min_common_intersection = np.clip(
            left["min_common_intersection"] + right["min_common_uncovered"] - unit_common,
            a_min=0,
            a_max=None,
        )
        max_common_intersection = np.minimum(right["max_common_uncovered"], left["max_common_intersection"])
        min_unique_extras = left["min_unique_extras"]
        max_unique_extras = left["max_unique_extras"]
        min_unique_extras_sum = left["min_unique_extras_sum"]
        max_unique_extras_sum = left["max_unique_extras_sum"]
        min_common_extras = np.clip(left["min_common_extras"] - right["max_common_extras"], a_min=0, a_max=None)
        max_common_extras = np.minimum(left["max_common_extras"], not_firing_common - right["min_common_extras"])

    elif isinstance(formula, F.And):
        max_unique_intersection = np.zeros_like(left["min_unique_intersection"])
        min_unique_intersection = np.zeros_like(left["min_unique_intersection"])
        min_unique_extras = np.zeros_like(left["min_unique_extras"])
        max_unique_extras = np.zeros_like(left["min_unique_extras"])
        max_unique_intersection_sum = 0
        min_unique_intersection_sum = 0
        min_unique_extras_sum = 0
        max_unique_extras_sum = 0
        min_common_intersection = np.clip(
            left["min_common_intersection"] + right["min_common_intersection"] - unit_common,
            a_min=0,
            a_max=None,
        )
        max_common_intersection = np.minimum(left["max_common_intersection"], right["max_common_intersection"])
        min_common_extras = np.clip(
            left["min_common_extras"] + right["min_common_extras"] - not_firing_common,
            a_min=0,
            a_max=None,
        )
        max_common_extras = np.minimum(left["max_common_extras"], right["max_common_extras"])

    else:
        raise ValueError(f"Unknown formula type: {type(formula)}")

    return {
        "max_common_intersection": max_common_intersection,
        "min_common_intersection": min_common_intersection,
        "max_unique_intersection": max_unique_intersection,
        "min_unique_intersection": min_unique_intersection,
        "max_common_extras": max_common_extras,
        "min_common_extras": min_common_extras,
        "max_unique_extras": max_unique_extras,
        "min_unique_extras": min_unique_extras,
        "max_common_intersection_sum": max_common_intersection.sum(),
        "min_common_intersection_sum": min_common_intersection.sum(),
        "max_unique_intersection_sum": max_unique_intersection_sum,
        "min_unique_intersection_sum": min_unique_intersection_sum,
        "max_common_extras_sum": max_common_extras.sum(),
        "min_common_extras_sum": min_common_extras.sum(),
        "max_unique_extras_sum": max_unique_extras_sum,
        "min_unique_extras_sum": min_unique_extras_sum,
    }


def are_disjoint(formula_a: F.F, formula_b: F.F, disjoint_info: np.ndarray) -> bool:
    """Returns whether the masks of the two formulas never overlap. Only the
    case where formula_b is a concept or a negated concept is handled.

    Args:
        formula_a (F.F): first formula.
        formula_b (F.F): second formula.
        disjoint_info (np.ndarray): disjoint matrix of the concepts.
    """
    if isinstance(formula_b, F.Not):
        return are_disjoint(formula_a, formula_b.val, disjoint_info)
    if not isinstance(formula_b, F.Leaf):
        return False
    concept_b = formula_b.val

    if isinstance(formula_a, F.Leaf):
        return bool(disjoint_info[formula_a.val, concept_b])
    if isinstance(formula_a, F.Or):
        # An OR is disjoint from b only if both operands are
        left_disjoint = are_disjoint(formula_a.left, formula_b, disjoint_info)
        right_disjoint = are_disjoint(formula_a.right, formula_b, disjoint_info)
        return left_disjoint and right_disjoint
    if isinstance(formula_a, F.And):
        if isinstance(formula_a.right, F.Not):
            # L AND NOT c is contained in L
            return are_disjoint(formula_a.left, formula_b, disjoint_info)
        left_disjoint = are_disjoint(formula_a.left, formula_b, disjoint_info)
        right_disjoint = are_disjoint(formula_a.right, formula_b, disjoint_info)
        return left_disjoint or right_disjoint
    raise ValueError(f"Unknown formula type: {type(formula_a)}")


def combine_disjoint_quantities(formula: F.F, *, left_quantities: dict, right_quantities: dict) -> Optional[dict]:
    """Estimates the quantities of the formula from the ones of its operands,
    when the operands never overlap.

    Returns:
        dict: estimated quantities, or None if the formula is useless.
    """
    left, right = left_quantities, right_quantities

    # Discard the formula if one of the operands never intersects the unit
    right_never_intersects = (
        right["max_unique_intersection_sum"] == 0
        and right["min_unique_intersection_sum"] == 0
        and right["max_common_intersection_sum"] == 0
        and right["min_common_intersection_sum"] == 0
    )
    left_never_intersects = (
        left["max_unique_intersection_sum"] == 0
        and left["min_unique_intersection_sum"] == 0
        and left["max_common_intersection_sum"] == 0
        and left["min_common_intersection_sum"] == 0
    )
    if right_never_intersects or left_never_intersects:
        return None

    if isinstance(formula, F.Or):
        quantities = {
            "max_common_intersection": left["max_common_intersection"] + right["max_common_intersection"],
            "min_common_intersection": left["min_common_intersection"] + right["min_common_intersection"],
            "max_unique_intersection": left["max_unique_intersection"] + right["max_unique_intersection"],
            "min_unique_intersection": left["min_unique_intersection"] + right["min_unique_intersection"],
            "max_common_extras": left["max_common_extras"] + right["max_common_extras"],
            "min_common_extras": left["min_common_extras"] + right["min_common_extras"],
            "max_unique_extras": left["max_unique_extras"] + right["max_unique_extras"],
            "min_unique_extras": left["min_unique_extras"] + right["min_unique_extras"],
        }
        # Sum of each quantity over the samples
        for name in list(quantities):
            quantities[f"{name}_sum"] = quantities[name].sum()
        return quantities
    if isinstance(formula, F.And):
        # L AND c is empty and L AND NOT c is equal to L
        return None
    raise ValueError(f"Unknown formula type: {type(formula)}")


def estimate_quantities(
    formula: F.F,
    *,
    label_mapping: dict,
    concepts_quantities: list,
    neuron_quantities: tuple,
    disjoint_info: np.ndarray,
) -> Optional[dict]:
    """Returns the exact or estimated quantities of the formula.

    Args:
        formula (F.F): the formula.
        label_mapping (dict): index of each formula with known quantities.
        concepts_quantities (list): known quantities.
        neuron_quantities (tuple): quantities of the unit.
        disjoint_info (np.ndarray): disjoint matrix of the concepts.

    Returns:
        dict: quantities of the formula, or None if the formula is discarded.
    """
    if len(formula) == 1 or formula in label_mapping:
        return concepts_quantities[label_mapping[formula]]

    # The quantities of NOT c are derived from the ones of c
    left_operand = formula.left.val if isinstance(formula.left, F.Not) else formula.left
    right_operand = formula.right.val if isinstance(formula.right, F.Not) else formula.right

    left_quantities = estimate_quantities(
        left_operand,
        label_mapping=label_mapping,
        concepts_quantities=concepts_quantities,
        neuron_quantities=neuron_quantities,
        disjoint_info=disjoint_info,
    )
    if left_quantities is None:
        return None
    right_quantities = estimate_quantities(
        right_operand,
        label_mapping=label_mapping,
        concepts_quantities=concepts_quantities,
        neuron_quantities=neuron_quantities,
        disjoint_info=disjoint_info,
    )
    if right_quantities is None:
        return None

    if are_disjoint(formula.left, formula.right, disjoint_info):
        return combine_disjoint_quantities(
            formula, left_quantities=left_quantities, right_quantities=right_quantities
        )
    return combine_quantities(
        formula,
        left_quantities=left_quantities,
        right_quantities=right_quantities,
        neuron_quantities=neuron_quantities,
    )


def estimate_max_iou(formula: F.F, *, label_mapping: dict, unit_heuristic_info: dict, num_hits: int) -> float:
    """Returns an upper bound of the IoU of the formula.

    Args:
        formula (F.F): the formula.
        label_mapping (dict): index of each formula with known quantities.
        unit_heuristic_info (dict): information of the
            unit used by the heuristic.
        num_hits (int): number of pixels of the unit mask.

    Returns:
        float: upper bound of the IoU, or 0 if the formula is discarded.
    """
    quantities = estimate_quantities(
        formula,
        label_mapping=label_mapping,
        concepts_quantities=unit_heuristic_info["concepts_quantities"],
        neuron_quantities=unit_heuristic_info["neuron_quantities"],
        disjoint_info=unit_heuristic_info["disjoint_info"],
    )
    if quantities is None:
        return 0.0

    # Largest intersection over smallest union
    max_intersection = quantities["max_common_intersection_sum"] + quantities["max_unique_intersection_sum"]
    if max_intersection == 0:
        return 0.0
    min_union = num_hits + quantities["min_unique_extras_sum"] + quantities["min_common_extras_sum"]
    return max_intersection / min_union
