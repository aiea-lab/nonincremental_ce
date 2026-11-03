"""Estimators of the IoU of the candidate formulas."""

import torch

from compositional import formula as F
from compositional import heuristic, heuristic_info, mmesh_heuristic
from utils import mask_utils


class Estimator:
    """Base class of the estimators of the IoU of the candidate formulas."""

    def estimate_max_iou(self, formula: F.F) -> float:
        """Returns an upper bound of the IoU of the formula."""
        raise NotImplementedError

    def add_beam_formulas(self, beam_formulas: list, *, beam_masks: dict):
        """Computes the masks and the information of the beam formulas.

        Args:
            beam_formulas (list): formulas of the beam.
            beam_masks (dict): masks of the beam formulas, updated in place.
        """
        raise NotImplementedError


class QuantitiesEstimator(Estimator):
    """Estimator of the IoU of the candidate formulas based on the quantities
    of the heuristic."""

    def __init__(self, *, masks: list, bitmaps: torch.Tensor, unit_heuristic_info: dict, max_length: int):
        """
        Args:
            masks (list): sparse concept masks of shape (num_samples, H*W).
            bitmaps (torch.Tensor): boolean unit mask of shape (num_samples, H*W).
            unit_heuristic_info (dict): information of the
                unit used by the heuristic.
            max_length (int): maximum length of the explanations.
        """
        self.masks = masks
        self.bitmaps = bitmaps
        self.unit_heuristic_info = unit_heuristic_info
        self.max_length = max_length
        self.num_hits = int(bitmaps.sum().item())
        # Index of the quantities of each formula
        self.label_mapping = {F.Leaf(concept): concept for concept in range(len(masks))}

    def estimate_max_iou(self, formula: F.F) -> float:
        """Returns an upper bound of the IoU of the formula."""
        return heuristic.estimate_max_iou(
            formula,
            label_mapping=self.label_mapping,
            unit_heuristic_info=self.unit_heuristic_info,
            num_hits=self.num_hits,
        )

    def add_beam_formulas(self, beam_formulas: list, *, beam_masks: dict):
        """Computes the masks and the quantities of the beam formulas.

        Args:
            beam_formulas (list): formulas of the beam.
            beam_masks (dict): masks of the beam formulas, updated in place.
        """
        concepts_quantities = self.unit_heuristic_info["concepts_quantities"]
        device = self.bitmaps.device
        common_elements, unique_elements, uncoverable_elements = self.unit_heuristic_info["dataset_quantities"]
        device_quantities = (
            common_elements.to(device),
            unique_elements.to(device),
            uncoverable_elements.to(device),
        )

        new_quantities = {}
        for formula in beam_formulas:
            if formula in beam_masks or isinstance(formula, F.Leaf):
                continue
            formula_mask = mask_utils.get_formula_mask(formula, self.masks, beam_masks, device=device)
            new_quantities[formula] = heuristic_info.compute_concept_quantities(
                formula_mask,
                bitmaps=self.bitmaps,
                dataset_quantities=device_quantities,
                neuron_quantities=self.unit_heuristic_info["neuron_quantities"],
            )
            beam_masks[formula] = formula_mask.cpu()

        for formula, formula_quantities in new_quantities.items():
            if len(formula) >= self.max_length:
                # Formulas of maximum length are never expanded
                continue
            if formula not in self.label_mapping:
                self.label_mapping[formula] = len(self.label_mapping)
                concepts_quantities.append(formula_quantities)
            else:
                concepts_quantities[self.label_mapping[formula]] = formula_quantities


class MMESHEstimator(Estimator):
    """Estimator of the IoU of the candidate formulas based on MMESH."""

    def __init__(
        self,
        *,
        masks: list,
        bitmaps: torch.Tensor,
        unit_heuristic_info: dict,
        max_length: int,
        concepts_info: dict,
        mask_shape: tuple,
    ):
        """
        Args:
            masks (list): sparse concept masks of shape (num_samples, H*W).
            bitmaps (torch.Tensor): boolean unit mask of shape (num_samples, H*W).
            unit_heuristic_info (dict): information of the
                unit used by the heuristic.
            max_length (int): maximum length of the explanations.
            concepts_info (dict): MMESH information of the concepts.
            mask_shape (tuple): shape (H, W) of the mask of a sample.
        """
        self.masks = masks
        self.bitmaps = bitmaps
        self.max_length = max_length
        self.num_hits = int(bitmaps.sum().item())

        # Intersections of the concepts with the unit, taken from the quantities
        # of the heuristic
        concepts_quantities = unit_heuristic_info["concepts_quantities"]
        num_concepts = concepts_info["areas"].shape[0]
        concepts_intersections = []
        for quantities in concepts_quantities[:num_concepts]:
            common = quantities["max_common_intersection"].astype("int32")
            unique = quantities["max_unique_intersection"].astype("int32")
            concepts_intersections.append(torch.from_numpy(common + unique))

        device = bitmaps.device
        self.unit_info = {
            "concepts": {key: value.to(device) for key, value in concepts_info.items()},
            "neuron_areas": bitmaps.sum(dim=1, dtype=torch.int32),
            "concepts_intersections": torch.stack(concepts_intersections).to(device),
            "mask_shape": mask_shape,
            "max_area": mask_shape[0] * mask_shape[1],
            "formulas": {},
        }

    def estimate_max_iou(self, formula: F.F) -> float:
        """Returns the MMESH estimate of the IoU of the formula."""
        return mmesh_heuristic.estimate_max_iou(formula, unit_info=self.unit_info, num_hits=self.num_hits)

    def add_beam_formulas(self, beam_formulas: list, *, beam_masks: dict):
        """Computes the masks and the MMESH information of the beam formulas.

        Args:
            beam_formulas (list): formulas of the beam.
            beam_masks (dict): masks of the beam formulas, updated in place.
        """
        device = self.bitmaps.device
        for formula in beam_formulas:
            if formula in beam_masks or isinstance(formula, F.Leaf):
                continue
            formula_mask = mask_utils.get_formula_mask(formula, self.masks, beam_masks, device=device)
            beam_masks[formula] = formula_mask.cpu()
            if len(formula) >= self.max_length:
                # Formulas of maximum length are never expanded
                continue
            formula_info = mmesh_heuristic.compute_masks_info(formula_mask, mask_shape=self.unit_info["mask_shape"])
            formula_info["intersections"] = (formula_mask & self.bitmaps).sum(dim=1, dtype=torch.int32)
            self.unit_info["formulas"][formula] = formula_info
