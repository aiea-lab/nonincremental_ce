"""Script to compute compositional explanations of neurons."""

import os
import time
from typing import Callable, Optional

import torch
import absl.flags
import absl.app
from tqdm import tqdm

from src import common_flags, settings  # common_flags registers all user flags
from compositional import formula as F
from compositional import estimators, heuristic_info, incremental_beam_search, non_incremental_beam_search
from compositional import metrics
from src import annotations, constants
from utils import activation_utils, common_utils, concepts_utils, mask_utils, model_utils, search_utils

FLAGS = absl.flags.FLAGS


def explain_units(*, search_name: str, beam_search: Optional[Callable], search_args: dict):
    """Computes the explanations of the selected units and prints them. The
    explanations already stored are loaded.

    Args:
        search_name (str): name of the search, used to store the results.
        beam_search (callable): beam search function with the signature of
            incremental_beam_search.incremental_beam_search; None for the
            optimal search.
        search_args (dict): arguments of the beam search besides the ones
            shared by all the searches.
    """
    # Set the seed
    common_utils.set_seed(FLAGS.seed)

    # Settings of the run
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

    # Load the model
    model = model_utils.load_model(cfg)

    # Load the activations
    print(
        f"Loading activations for {cfg.model} model on {cfg.dataset} dataset - Layer:{cfg.layer}"
    )
    activation_file = cfg.get_activations_file_path()
    layer_activations = model.get_layer_activations(layer=cfg.layer, layer_file=activation_file)
    print(f"Activations loaded. Shape: {tuple(layer_activations.shape)}")

    # Select the units
    selected_units = model_utils.get_selected_units(
        layer_activations, units=FLAGS.units, random_units=FLAGS.random_units, seed=FLAGS.seed
    )
    selected_units = sorted(selected_units)
    print(f"Selected {len(selected_units)} units")

    # Load the concepts
    if cfg.get_dataset_name() == "snli":
        masks, concept_labels, constraints = concepts_utils.get_nli_concepts(cfg, model.data_loader)
    else:
        broden_annotations = annotations.BrodenAnnotationsWrapper(cfg)
        masks = broden_annotations.get_masks(cfg.get_segmentations_dir())
        concept_labels = broden_annotations.concept_labels
        constraints = None
    disjoint_info = mask_utils.disjoint_matrix_from_masks(masks)
    dataset_quantities = concepts_utils.get_dataset_quantities(masks, cfg.get_info_dir())
    print(f"Concepts loaded. Number of concepts: {len(masks)}")

    if beam_search is None:
        beam_limit = None
    else:
        if FLAGS.beam_limit is None:
            raise absl.app.UsageError("--beam_limit is required by the beam searches")
        beam_limit = FLAGS.beam_limit
        if cfg.get_dataset_name() == "snli":
            first_beam_size = beam_limit
        else:
            first_beam_size = beam_limit * constants.FIRST_BEAM_FACTOR

    # Explain each unit
    for unit in tqdm(selected_units, desc="Explaining units", unit="unit"):
        unit_activations = layer_activations[:, unit]
        if not activation_utils.has_enough_activations(unit_activations):
            tqdm.write(f"Unit {unit}: too few non-zero activations, skipped")
            continue

        threshold = activation_utils.compute_threshold(unit_activations)
        bitmaps = activation_utils.compute_bitmaps(
            unit_activations, threshold, cfg.get_mask_shape()
        )
        bitmaps = bitmaps.to(cfg.device)
        num_active_pixels = bitmaps.sum().item()
        if num_active_pixels == 0:
            tqdm.write(f"Unit {unit}: empty activation mask, skipped")
            continue

        results_path = common_utils.get_unit_results_path(
            cfg.get_results_dir(),
            search_name=search_name,
            beam_limit=beam_limit,
            length=FLAGS.length,
            unit=unit,
            threshold=threshold,
        )
        if os.path.exists(results_path) and not FLAGS.regenerate:
            unit_results = common_utils.load_unit_results(results_path)
        else:
            # The time includes the heuristic info and the search
            start_time = time.time()
            if beam_search is None:
                # Imported here, so that the beam searches do not need the package
                from compositional_explanations import optimal

                optimal_formula = optimal.compute_optimal_explanations(
                    masks=masks,
                    bitmaps=bitmaps,
                    disjoint_info=disjoint_info,
                    length=FLAGS.length,
                    device=cfg.device,
                    cache_dir=cfg.get_info_dir(),
                )
                best_formula = F.from_optimal_formula(optimal_formula)
                best_iou = search_utils.compute_iou(best_formula, masks, bitmaps)
                visited, expanded, estimated = None, None, None
            else:
                # Quantities used by the heuristic of the beam search
                unit_heuristic_info = heuristic_info.compute_heuristic_info(
                    masks, bitmaps, dataset_quantities=dataset_quantities, disjoint_info=disjoint_info
                )
                best_formula, best_iou, visited, expanded, estimated = beam_search(
                    masks,
                    bitmaps,
                    unit_heuristic_info,
                    beam_size=beam_limit,
                    length=FLAGS.length,
                    first_beam_size=first_beam_size,
                    constraints=constraints,
                    **search_args,
                )
            elapsed_time = time.time() - start_time

            # Quality of the explanation
            formula_mask = mask_utils.get_formula_mask(best_formula, masks, device=bitmaps.device)
            unit_results = {
                "formula": best_formula,
                "formula_str": F.get_formula_str(best_formula, concept_labels),
                "iou": best_iou,
                "detection_accuracy": metrics.detection_accuracy(
                    formula_mask=formula_mask, unit_mask=bitmaps
                ),
                "activation_coverage": metrics.activation_coverage(
                    formula_mask=formula_mask, unit_mask=bitmaps
                ),
                "visited": visited,
                "expanded": expanded,
                "estimated": estimated,
                "time": elapsed_time,
            }
            common_utils.save_unit_results(results_path, unit_results)

        tqdm.write(
            f"Unit {unit}: {unit_results['formula_str']} | "
            f"IoU {unit_results['iou']:.4f}, DetAcc {unit_results['detection_accuracy']:.4f}, "
            f"ActCov {unit_results['activation_coverage']:.4f} | visited {unit_results['visited']}, "
            f"expanded {unit_results['expanded']}, "
            f"estimated {unit_results['estimated']}, {unit_results['time']:.1f} s"
        )


def main(argv):
    search_args = {"estimator_class": estimators.QuantitiesEstimator, "estimator_args": {}}
    if FLAGS.search == "incremental":
        beam_search = incremental_beam_search.incremental_beam_search
    elif FLAGS.search == "non_incremental":
        beam_search = non_incremental_beam_search.non_incremental_beam_search
    else:
        # The optimal search does not use the beam search functions
        beam_search = None
    explain_units(search_name=FLAGS.search, beam_search=beam_search, search_args=search_args)


if __name__ == "__main__":
    with torch.no_grad():
        absl.flags.mark_flags_as_required(["model", "dataset", "layer", "search", "length"])
        absl.app.run(main)
