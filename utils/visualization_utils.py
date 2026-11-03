"""Utilities to visualize the samples explained by the explanation of a unit."""
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision

from src import segmentations


# ============================================================================
# VISION
# ============================================================================


def select_samples(explained_activations: torch.Tensor, *, num_samples: int) -> list:
    """Returns the samples with the most pixels where the unit is active and
    the explanation holds.

    Args:
        explained_activations (torch.Tensor): boolean mask of shape
            (num_samples, H*W) of the pixels where the unit is active and the
            explanation holds.
        num_samples (int): maximum number of samples to select.

    Returns:
        list: indices of the selected samples, sorted by decreasing coverage.
    """
    num_pixels = explained_activations.shape[1]
    coverage = explained_activations.sum(dim=1) / num_pixels
    sorted_coverage, sorted_indices = torch.sort(coverage, descending=True)
    covered_indices = sorted_indices[sorted_coverage > 0]
    return covered_indices[:num_samples].tolist()


def draw_samples(
    dataset: segmentations.BrodenDataset,
    sample_indices: list,
    explained_activations: torch.Tensor,
    *,
    mask_shape: tuple,
) -> torch.Tensor:
    """Draws in blue the pixels where the unit is active and the explanation
    holds on the images of the samples.

    Args:
        dataset (BrodenDataset): dataset of the images.
        sample_indices (list): indices of the samples to draw.
        explained_activations (torch.Tensor): boolean mask of shape
            (num_samples, H*W) of the pixels where the unit is active and the
            explanation holds.
        mask_shape (tuple): shape (H, W) of the masks. The images are resized
            to this shape.

    Returns:
        torch.Tensor: grid of the drawn images, of shape (3, H', W').
    """
    drawn_images = []
    for index in sample_indices:
        image = torchvision.io.read_image(
            dataset.get_image_path(index), mode=torchvision.io.ImageReadMode.RGB
        )
        image = torchvision.transforms.functional.resize(image, mask_shape)
        explained_pixels = explained_activations[index].reshape(mask_shape).cpu()
        drawn_image = torchvision.utils.draw_segmentation_masks(
            image, explained_pixels, alpha=0.6, colors="blue"
        )
        drawn_images.append(drawn_image)
    return torchvision.utils.make_grid(drawn_images, padding=2, pad_value=255)


def save_comparison_figure(grids: list, *, titles: list, path: str):
    """Saves a figure with one row per grid of samples, each with its title.

    Args:
        grids (list): grids of images returned by draw_samples.
        titles (list): title of each row.
        path (str): path of the image file.
    """
    plt.rcParams["font.family"] = "DejaVu Sans"
    figure, axes = plt.subplots(
        nrows=len(grids), squeeze=False, gridspec_kw={"wspace": 0, "hspace": 0.5}
    )
    figure.set_figwidth(8.5)
    figure.set_figheight(4.2)
    for row, (grid, title) in enumerate(zip(grids, titles)):
        image = torchvision.transforms.functional.to_pil_image(grid)
        axes[row, 0].imshow(np.asarray(image))
        axes[row, 0].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        axes[row, 0].set_title(title)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    figure.savefig(path)
    plt.close(figure)


# ============================================================================
# NATURAL LANGUAGE INFERENCE
# ============================================================================


def select_top_activating_samples(
    *, unit_activations: torch.Tensor, explained_activations: torch.Tensor, num_samples: int
) -> list:
    """Returns the samples with the highest activations among the ones where
    the unit is active and the explanation holds.

    Args:
        unit_activations (torch.Tensor): activations of the unit, of shape
            (num_samples,).
        explained_activations (torch.Tensor): boolean mask of shape
            (num_samples, 1) of the samples where the unit is active and the
            explanation holds.
        num_samples (int): maximum number of samples to select.

    Returns:
        list: indices of the selected samples, sorted by decreasing activation.
    """
    explained_samples = torch.nonzero(explained_activations.squeeze(1).cpu()).squeeze(1)
    explained_sample_activations = unit_activations[explained_samples]
    order = torch.argsort(explained_sample_activations, descending=True)
    return explained_samples[order][:num_samples].tolist()


def save_nli_comparison(path: str, *, unit: int, explanations: list):
    """Saves a text file comparing the explanations of a unit, each followed by
    the premise-hypothesis pairs it selects.

    Args:
        path (str): path of the text file.
        unit (int): index of the unit.
        explanations (list): dicts with the keys "title", "formula_str", "iou",
            "iou_decimals", "other_title", and "pairs". The titles are the
            names of the two searches. Each pair is a tuple of premise,
            hypothesis, label, and whether the other explanation holds on it.
    """
    lines = [f"Unit {unit}", ""]
    for explanation in explanations:
        lines.append(f"{explanation['title']} - IoU: {explanation['iou']:.{explanation['iou_decimals']}f}")
        lines.append(f"Explanation: {explanation['formula_str']}")
        lines.append(f"Samples selected by this explanation ({explanation['other_title']} applies?):")
        for premise, hypothesis, label, other_applies in explanation["pairs"]:
            lines.append(f"  P: {premise}")
            lines.append(f"  H: {hypothesis}")
            lines.append(f"     label: {label} | {explanation['other_title']} applies: {'yes' if other_applies else 'no'}")
        lines.append("")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as text_file:
        text_file.write("\n".join(lines))
