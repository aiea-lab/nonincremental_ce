"""Script to compute the explanations of the non-incremental beam search
without one of its components."""

import absl.app
import absl.flags
import torch

import main
from compositional import estimators, non_incremental_beam_search

absl.flags.DEFINE_enum(
    "ablation",
    None,
    ["no_recombinations", "no_functional_equivalence", "no_non_incremental_expansion"],
    "component removed from the non-incremental beam search",
)

FLAGS = absl.flags.FLAGS

# Argument of non_incremental_beam_search that removes each component
ABLATION_ARGUMENTS = {
    "no_recombinations": {"use_recombinations": False},
    "no_functional_equivalence": {"use_functional_equivalence": False},
    "no_non_incremental_expansion": {"use_non_incremental_expansion": False},
}


def run_ablation(argv):
    search_args = {
        "estimator_class": estimators.QuantitiesEstimator,
        "estimator_args": {},
        **ABLATION_ARGUMENTS[FLAGS.ablation],
    }
    main.explain_units(
        search_name=FLAGS.ablation,
        beam_search=non_incremental_beam_search.non_incremental_beam_search,
        search_args=search_args,
    )


if __name__ == "__main__":
    with torch.no_grad():
        absl.flags.mark_flags_as_required(["model", "dataset", "layer", "ablation", "length", "beam_limit"])
        absl.app.run(run_ablation)
