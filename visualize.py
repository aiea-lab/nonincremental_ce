"""Script to visualize the explanations of the incremental and non-incremental
beam searches on the units where they differ."""

import os

import absl.app
import absl.flags
import torch

from src import annotations, common_flags, constants, settings  # common_flags registers all user flags
from src import segmentations, snli
from utils import activation_utils, common_utils, concepts_utils, mask_utils, model_utils, visualization_utils

absl.flags.DEFINE_string("root_figures", "data/figures", "root directory of the figures")

FLAGS = absl.flags.FLAGS

# Searches shown for each unit, in this order, with their names
VISUALIZED_SEARCHES = {"non_incremental": "Ours", "incremental": "Incremental"}


def get_iou_decimals(ious: list, *, min_decimals: int = 4) -> int:
    """Returns the number of decimals that prints different IoUs as different
    numbers.

    Args:
        ious (list): IoUs to print.
        min_decimals (int): minimum number of decimals.
    """
    decimals = min_decimals
    while len({round(iou, decimals) for iou in ious}) < len(set(ious)):
        decimals += 1
    return decimals


def save_vision_figure(
    path: str, *, unit: int, explanations: list, dataset: segmentations.BrodenDataset, mask_shape: tuple
):
    """Saves the figure of a vision unit with one row of images per explanation."""
    grids = []
    titles = []
    for explanation in explanations:
        sample_indices = visualization_utils.select_samples(
            explanation["explained_activations"], num_samples=constants.VISUALIZED_SAMPLES
        )
        grids.append(
            visualization_utils.draw_samples(
                dataset, sample_indices, explanation["explained_activations"], mask_shape=mask_shape
            )
        )
        optimal_mark = r" ($\checkmark$)" if explanation["is_optimal"] else ""
        titles.append(
            f"{explanation['title']}{optimal_mark}: {explanation['formula_str']} | IoU: {explanation['iou']:.{explanation['iou_decimals']}f}"
        )
    titles[0] = f"Unit: {unit}\n{titles[0]}"
    visualization_utils.save_comparison_figure(grids, titles=titles, path=path)


def save_nli_text(
    path: str, *, unit: int, explanations: list, unit_activations: torch.Tensor, dataset: snli.AnalysisDataset
):
    """Saves the text file of an NLI unit with the pairs selected by each explanation."""
    for explanation, other_explanation in zip(explanations, reversed(explanations)):
        sample_indices = visualization_utils.select_top_activating_samples(
            unit_activations=unit_activations,
            explained_activations=explanation["explained_activations"],
            num_samples=constants.VISUALIZED_SENTENCE_PAIRS,
        )
        pairs = []
        for index in sample_indices:
            premise, hypothesis, label = dataset.get_pair_text(index)
            other_applies = bool(other_explanation["formula_mask"][index, 0])
            pairs.append((premise, hypothesis, label, other_applies))
        explanation["pairs"] = pairs
        explanation["other_title"] = other_explanation["title"]
    visualization_utils.save_nli_comparison(path, unit=unit, explanations=explanations)


def main(argv):
    cfg = settings.Settings(
        model=FLAGS.model,
        dataset=FLAGS.dataset,
        layer=FLAGS.layer,
        device=FLAGS.device,
        root_models=FLAGS.root_models,
        root_datasets=FLAGS.root_datasets,
        root_activations=FLAGS.root_activations,
        root_segmentations=FLAGS.root_segmentations,
        step_size=FLAGS.step_size,
        fast_impl=FLAGS.fast_impl,
        root_results=FLAGS.root_results,
    )
    results_dir = cfg.get_results_dir()

    # Explanations of each search
    all_results = {}
    for search_name in VISUALIZED_SEARCHES:
        all_results[search_name] = common_utils.load_search_results(
            results_dir, search_name=search_name, beam_limit=FLAGS.beam_limit, length=FLAGS.length
        )
    optimal_results = common_utils.load_search_results(
        results_dir, search_name="optimal", beam_limit=None, length=FLAGS.length
    )

    # Units where the two explanations have a different IoU
    common_units = set(all_results["incremental"]) & set(all_results["non_incremental"])
    if FLAGS.units is not None:
        common_units &= {int(unit) for unit in FLAGS.units}
    units = [
        unit
        for unit in sorted(common_units)
        if all_results["incremental"][unit]["iou"] != all_results["non_incremental"][unit]["iou"]
    ]
    print(f"{cfg.model} {cfg.layer}: the explanations differ on {len(units)} of {len(common_units)} units")
    if len(units) == 0:
        return

    # Activations, concept masks, and samples
    model = model_utils.load_model(cfg)
    layer_activations = model.get_layer_activations(layer=cfg.layer, layer_file=cfg.get_activations_file_path())
    if cfg.get_dataset_name() == "snli":
        masks, _, _ = concepts_utils.get_nli_concepts(cfg, model.data_loader)
        dataset = model.data_loader.dataset
    else:
        broden_annotations = annotations.BrodenAnnotationsWrapper(cfg)
        masks = broden_annotations.get_masks(cfg.get_segmentations_dir())
        dataset = broden_annotations.data_loader.dataset
    mask_shape = cfg.get_mask_shape()

    figures_dir = os.path.join(FLAGS.root_figures, cfg.dataset, cfg.model, cfg.layer)
    for unit in units:
        unit_activations = layer_activations[:, unit]
        threshold = activation_utils.compute_threshold(unit_activations)
        bitmaps = activation_utils.compute_bitmaps(unit_activations, threshold, mask_shape)
        bitmaps = bitmaps.to(cfg.device)

        # The IoUs are printed with at least 4 decimals in vision and 3 in NLI
        min_decimals = 3 if cfg.get_dataset_name() == "snli" else 4
        iou_decimals = get_iou_decimals(
            [all_results[search_name][unit]["iou"] for search_name in VISUALIZED_SEARCHES],
            min_decimals=min_decimals,
        )
        # Explanation of each search and the samples where it holds
        explanations = []
        for search_name, search_title in VISUALIZED_SEARCHES.items():
            unit_results = all_results[search_name][unit]
            formula_mask = mask_utils.get_formula_mask(unit_results["formula"], masks, device=bitmaps.device)
            is_optimal = unit in optimal_results and unit_results["iou"] == optimal_results[unit]["iou"]
            explanations.append(
                {
                    "title": search_title,
                    "formula_str": unit_results["formula_str"],
                    "iou": unit_results["iou"],
                    "iou_decimals": iou_decimals,
                    "is_optimal": is_optimal,
                    "formula_mask": formula_mask,
                    "explained_activations": bitmaps & formula_mask,
                }
            )

        if cfg.get_dataset_name() == "snli":
            path = os.path.join(figures_dir, f"unit_{unit}.txt")
            save_nli_text(
                path, unit=unit, explanations=explanations, unit_activations=unit_activations, dataset=dataset
            )
        else:
            path = os.path.join(figures_dir, f"unit_{unit}.png")
            save_vision_figure(
                path, unit=unit, explanations=explanations, dataset=dataset, mask_shape=mask_shape
            )
        print(f"Unit {unit}: saved {path}")


if __name__ == "__main__":
    with torch.no_grad():
        absl.flags.mark_flags_as_required(["model", "dataset", "layer", "length", "beam_limit"])
        absl.app.run(main)
