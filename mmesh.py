"""Script to compute the explanations of the incremental beam search with the
MMESH heuristic."""

import absl.app
import absl.flags
import torch

import main
from compositional import estimators, incremental_beam_search, mmesh_heuristic
from src import annotations, settings

FLAGS = absl.flags.FLAGS


def run_mmesh(argv):
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
    if cfg.get_dataset_name() == "snli":
        raise absl.app.UsageError("MMESH uses the geometry of the masks, so it is defined only in vision")
    mask_shape = cfg.get_mask_shape()

    # MMESH information of the concepts, computed once for all the units
    masks = annotations.BrodenAnnotationsWrapper(cfg).get_masks(cfg.get_segmentations_dir())
    concepts_info = mmesh_heuristic.get_concepts_info(
        masks, mask_shape=mask_shape, info_dir=cfg.get_info_dir(), device=cfg.device
    )
    concepts_info = {key: value.to(cfg.device) for key, value in concepts_info.items()}
    del masks

    # Incremental beam search with the MMESH estimator
    search_args = {
        "estimator_class": estimators.MMESHEstimator,
        "estimator_args": {"concepts_info": concepts_info, "mask_shape": mask_shape},
    }
    main.explain_units(
        search_name="incremental_mmesh",
        beam_search=incremental_beam_search.incremental_beam_search,
        search_args=search_args,
    )


if __name__ == "__main__":
    with torch.no_grad():
        absl.flags.mark_flags_as_required(["model", "dataset", "layer", "length", "beam_limit"])
        absl.app.run(run_mmesh)
