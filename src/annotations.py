"""Wrappers of the annotations of the concept datasets. Adapted from
https://github.com/aiea-lab/optimal-compositional-explanations
"""

import os
from typing import Iterator

import scipy.sparse as sparse
import torch
from tqdm import tqdm

from src import constants, settings
from utils import dataset_utils


def resize_segmentations(segm_data: torch.Tensor, mask_shape: tuple) -> torch.Tensor:
    """Resizes the segmentations to the mask shape with nearest interpolation.

    Args:
        segm_data (torch.Tensor): segmentations of shape (B, C, H, W).
        mask_shape (tuple): shape of the masks.

    Returns:
        torch.Tensor: segmentations of shape (B, C, *mask_shape).
    """
    return torch.nn.functional.interpolate(
        segm_data.float(), size=mask_shape, mode="nearest"
    ).long()


def parse_concepts(resized_segm: torch.Tensor, concepts: list) -> torch.Tensor:
    """Returns the binary masks of the given concepts.

    Args:
        resized_segm (torch.Tensor): segmentations of shape (B, C, H, W).
        concepts (list): indices of the concepts.

    Returns:
        torch.Tensor: boolean masks of shape (B, len(concepts), H, W).
    """
    concepts = torch.tensor(concepts, device=resized_segm.device)
    return (resized_segm.unsqueeze(1) == concepts.view(1, -1, 1, 1, 1)).any(dim=2)


def load_sparse_masks(concept_names: list, segmentations_directory: str) -> list:
    """Loads the sparse masks of the given concepts.

    Returns:
        list: list of scipy sparse matrices of shape (num_samples, H*W).
    """
    return [
        sparse.load_npz(f"{segmentations_directory}/{concept}.npz")
        for concept in tqdm(concept_names, desc="Loading Sparse Masks")
    ]


class AnnotationsWrapper:
    """Base wrapper of the annotations of a concept dataset."""

    def __init__(self, cfg: settings.Settings):
        """
        Args:
            cfg (settings.Settings): settings of the run.
        """
        self.data_loader = None
        self.concept_labels = None
        self.mask_shape = cfg.get_mask_shape()
        self.device = cfg.get_device()
        self.step_size = cfg.get_step_size()
        self.fast_impl = cfg.fast_impl_is_enabled()
        # Resized segmentations of the whole dataset, used with fast_impl
        self.dataset_segmentations = None

    def extract_segmentations(self, data: tuple) -> torch.Tensor:
        """Returns the annotations of a batch of the data loader.

        Args:
            data (tuple): batch of the data loader.

        Returns:
            torch.Tensor: annotations of shape (B, C, H, W).
        """
        raise NotImplementedError

    def iter_loader_segmentations(self) -> Iterator[torch.Tensor]:
        """Yields the segmentations of each batch of the data loader, resized
        to the mask shape."""
        for data in tqdm(self.data_loader, desc="Extracting Masks", leave=False):
            segmentations = self.extract_segmentations(data).to(self.device)
            yield resize_segmentations(segmentations, self.mask_shape)

    def iter_memory_segmentations(self) -> Iterator[torch.Tensor]:
        """Yields batches of the dataset segmentations resized to the mask
        shape. The first call keeps all the segmentations in memory, so the
        next calls do not read the data loader again."""
        if self.dataset_segmentations is None:
            self.dataset_segmentations = torch.cat(
                [batch.to(torch.int16).cpu() for batch in self.iter_loader_segmentations()]
            )
        for batch in torch.split(self.dataset_segmentations, constants.FAST_IMPL_BATCH_SIZE):
            yield batch.to(self.device)

    def extract_masks(self, concepts: list) -> torch.Tensor:
        """Extracts the masks of the given concepts.

        Returns:
            torch.Tensor: boolean masks of shape (num_samples, len(concepts), H*W).
        """
        num_samples = len(self.data_loader.dataset)
        num_pixels = self.mask_shape[0] * self.mask_shape[1]
        masks = torch.zeros((num_samples, len(concepts), num_pixels), dtype=torch.bool)
        if self.fast_impl:
            segmentation_batches = self.iter_memory_segmentations()
        else:
            segmentation_batches = self.iter_loader_segmentations()
        start = 0
        for resized_segm in segmentation_batches:
            batch_masks = parse_concepts(resized_segm, concepts).flatten(2).cpu()
            masks[start : start + batch_masks.shape[0]] = batch_masks
            start += batch_masks.shape[0]
        return masks

    def save_masks(self, segmentations_dir: str, concepts_to_save: list):
        """Extracts and saves the sparse masks of the given concepts."""
        os.makedirs(segmentations_dir, exist_ok=True)
        for start in tqdm(
            range(0, len(concepts_to_save), self.step_size), desc="Saving Masks"
        ):
            concepts = concepts_to_save[start : start + self.step_size]
            masks = self.extract_masks(concepts)
            for position, concept_index in enumerate(concepts):
                sparse.save_npz(
                    f"{segmentations_dir}/{self.concept_labels[concept_index]}.npz",
                    sparse.csr_matrix(masks[:, position].numpy()),
                )
            del masks

    def get_masks(self, segmentations_dir: str) -> list:
        """Returns the masks of all the concepts, extracting the missing ones.

        Returns:
            list: list of scipy sparse matrices of shape (num_samples, H*W).
        """
        missing = [
            index
            for index, concept in enumerate(self.concept_labels)
            if not os.path.exists(f"{segmentations_dir}/{concept}.npz")
        ]
        if len(missing) > 0:
            print(f"Missing {len(missing)} masks in {segmentations_dir}")
            self.save_masks(segmentations_dir, missing)

        masks = load_sparse_masks(self.concept_labels, segmentations_dir)
        # The empty label is a placeholder, so its mask is empty
        for i, concept in enumerate(self.concept_labels):
            if concept == "":
                masks[i] = sparse.csr_matrix(masks[i].shape, dtype=bool)
        return masks


class BrodenAnnotationsWrapper(AnnotationsWrapper):
    """Annotations of the Broden dataset (vision)."""

    def __init__(self, cfg: settings.Settings):
        """
        Args:
            cfg (settings.Settings): settings of the run.
        """
        super().__init__(cfg)
        self.data_loader = dataset_utils.get_broden_data(cfg, resolution=cfg.get_annotations_size())
        self.concept_labels = self.data_loader.dataset.labels

    def extract_segmentations(self, data: tuple) -> torch.Tensor:
        """Returns the segmentations of a batch of the Broden data loader.

        Args:
            data (tuple): batch of the data loader.

        Returns:
            torch.Tensor: segmentations of shape (B, C, H, W).
        """
        return data[1]
