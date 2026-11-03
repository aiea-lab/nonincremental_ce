"""Script to compare the runtime of the incremental and non-incremental beam
searches for different beam sizes and explanation lengths."""

import absl.app
import absl.flags
import numpy as np

from src import common_flags, settings  # common_flags registers all user flags
from utils import common_utils

absl.flags.DEFINE_list("sweep_beam_limits", ["10"], "beam sizes to compare")
absl.flags.DEFINE_list("sweep_lengths", ["5"], "maximum explanation lengths to compare")

FLAGS = absl.flags.FLAGS

SEARCHES = ["incremental", "non_incremental"]


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
        root_results=FLAGS.root_results,
    )
    results_dir = cfg.get_results_dir()
    configurations = [
        (int(beam_limit), int(length))
        for beam_limit in FLAGS.sweep_beam_limits
        for length in FLAGS.sweep_lengths
    ]

    # Results of each search in each configuration
    all_results = {}
    for beam_limit, length in configurations:
        for search_name in SEARCHES:
            all_results[(beam_limit, length, search_name)] = common_utils.load_search_results(
                results_dir, search_name=search_name, beam_limit=beam_limit, length=length
            )

    # Units with results for both searches in every configuration
    common_units = None
    for search_results in all_results.values():
        if common_units is None:
            common_units = set(search_results)
        else:
            common_units &= set(search_results)
    if FLAGS.units is not None:
        common_units &= {int(unit) for unit in FLAGS.units}
    units = sorted(common_units)
    if len(units) == 0:
        raise ValueError("No unit has results for all the configurations: run main.py first.")
    print(f"{cfg.model} {cfg.layer}: {len(units)} units with results for both searches "
          f"in every configuration\n")

    print(
        f"{'beam':>5} {'length':>6} | {'incremental s/u':>16} | {'median':>6} | "
        f"{'non-incremental s/u':>19} | {'median':>6} | ratio of means | ratio of medians"
    )
    for beam_limit, length in configurations:
        times = {}
        for search_name in SEARCHES:
            search_results = all_results[(beam_limit, length, search_name)]
            times[search_name] = [search_results[unit]["time"] for unit in units]
        means = {search_name: np.mean(search_times) for search_name, search_times in times.items()}
        medians = {search_name: np.median(search_times) for search_name, search_times in times.items()}
        print(
            f"{beam_limit:>5} {length:>6} | "
            f"{means['incremental']:>7.1f} ± {np.std(times['incremental']):>6.1f} | {medians['incremental']:>6.1f} | "
            f"{means['non_incremental']:>10.1f} ± {np.std(times['non_incremental']):>6.1f} | {medians['non_incremental']:>6.1f} | "
            f"{means['non_incremental'] / means['incremental']:>14.2f} | "
            f"{medians['non_incremental'] / medians['incremental']:>16.2f}"
        )


if __name__ == "__main__":
    absl.flags.mark_flags_as_required(["model", "dataset", "layer"])
    absl.app.run(main)
