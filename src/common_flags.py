"""Flags of the scripts. Importing this module registers them."""

import absl.flags

# Run

absl.flags.DEFINE_string(
    "dataset", None, "dataset to use. Accepted values: [broden, snli]"
)
absl.flags.DEFINE_string(
    "model",
    None,
    "model to use. Accepted values: [resnet18, alexnet, densenet161, bowman, bert, llama]",
)
absl.flags.DEFINE_string(
    "layer",
    None,
    "layer to explain. Accepted values: layer4 for resnet18, features for alexnet and "
    "densenet161, mlp for bowman, bert, and llama",
)
absl.flags.DEFINE_string("device", "cuda", "device where the model is stored")

absl.flags.DEFINE_integer("length", None, "maximum length of the explanations")
absl.flags.DEFINE_integer(
    "beam_limit", None, "number of formulas kept in the beam of the beam searches"
)
absl.flags.DEFINE_integer(
    "random_units", 0, "number of units extracted randomly with --seed when --units is not given"
)
absl.flags.DEFINE_list("units", None, "units to explain")
absl.flags.DEFINE_integer(
    "step_size",
    80,
    "number of concepts whose masks are extracted in a single step. Lower it to reduce the memory usage",
)
absl.flags.DEFINE_boolean(
    "fast_impl",
    False,
    "whether to load all the dataset segmentations in memory to extract the masks. It is faster "
    "but requires more memory",
)
absl.flags.DEFINE_string("root_models", "data/model", "root directory of the models")
absl.flags.DEFINE_string("root_datasets", "data/datasets", "root directory of the datasets")
absl.flags.DEFINE_string(
    "root_segmentations", "data/cache/segmentations", "root directory of the concept masks"
)
absl.flags.DEFINE_string(
    "root_activations", "data/cache/activations", "root directory of the activations"
)
absl.flags.DEFINE_string("root_results", "data/results", "root directory of the explanations")
absl.flags.DEFINE_boolean(
    "regenerate",
    False,
    "whether to recompute the explanations already stored in root_results",
)

# Reproducibility

absl.flags.DEFINE_integer("seed", 0, "seed of the random generators")

# Search

absl.flags.DEFINE_enum(
    "search",
    None,
    ["incremental", "non_incremental", "optimal"],
    "search to use",
)
