"""Script to compare the explanations of two beam searches and of the optimal
search on the same units."""

import absl.app
import absl.flags
import numpy as np

from src import common_flags, settings  # common_flags registers all user flags
from utils import common_utils

absl.flags.DEFINE_list(
    "compared_searches",
    ["incremental", "non_incremental"],
    "names of the two beam searches whose results are compared. Accepted values: [incremental, "
    "non_incremental, incremental_mmesh, no_recombinations, no_functional_equivalence, "
    "no_non_incremental_expansion]",
)

absl.flags.DEFINE_boolean(
    "include_optimal",
    True,
    "whether to compare the beam searches with the optimal search. If enabled, only the units "
    "with an optimal result are compared",
)

FLAGS = absl.flags.FLAGS


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
    if len(FLAGS.compared_searches) != 2:
        raise absl.app.UsageError("--compared_searches needs two searches")
    beam_searches = FLAGS.compared_searches
    first_search, second_search = beam_searches

    # Results of each search
    all_results = {}
    for search_name in beam_searches:
        all_results[search_name] = common_utils.load_search_results(
            results_dir, search_name=search_name, beam_limit=FLAGS.beam_limit, length=FLAGS.length
        )
    if FLAGS.include_optimal:
        optimal_results = common_utils.load_search_results(
            results_dir, search_name="optimal", beam_limit=None, length=FLAGS.length
        )
        if len(optimal_results) > 0:
            all_results["optimal"] = optimal_results

    # Units with results for every search
    common_units = set(all_results[first_search])
    for search_results in all_results.values():
        common_units &= set(search_results)
    if FLAGS.units is not None:
        common_units &= {int(unit) for unit in FLAGS.units}
    units = sorted(common_units)
    if len(units) == 0:
        raise ValueError("No unit has results for all the searches: run main.py first.")
    # Units where the beam searches find a different IoU. The explanation
    # metrics are averaged on these units.
    different_units = [
        unit for unit in units
        if all_results[first_search][unit]["iou"] != all_results[second_search][unit]["iou"]
    ]
    print(f"{cfg.model} {cfg.layer}: {len(units)} units with results for {', '.join(all_results)}; "
          f"the beam searches find a different IoU on {len(different_units)} of them\n")

    for search_name, search_results in all_results.items():
        ious = [search_results[unit]["iou"] for unit in units]
        # The optimal results converted from older runs have no time
        times = [search_results[unit]["time"] for unit in units if search_results[unit]["time"] is not None]
        print(f"{search_name}")

        if search_name in beam_searches:
            if "optimal" in all_results:
                # Units where the explanation reaches the IoU of the optimal one
                optimal_ious = [all_results["optimal"][unit]["iou"] for unit in units]
                above_optimal = [unit for unit, iou, optimal_iou in zip(units, ious, optimal_ious) if iou > optimal_iou]
                if len(above_optimal) > 0:
                    raise ValueError(
                        f"The {search_name} search finds a higher IoU than the optimal search for "
                        f"units {above_optimal}. This is not supposed to happen. Please check you setup."
                                           )
                num_optimal = sum(iou >= optimal_iou for iou, optimal_iou in zip(ious, optimal_ious))
                print(f"  Optimal:   {num_optimal / len(units):.2f}")
            # Units where the explanation is strictly better than the other beam search
            other_search = [name for name in beam_searches if name != search_name][0]
            other_ious = [all_results[other_search][unit]["iou"] for unit in units]
            num_better = sum(iou > other_iou for iou, other_iou in zip(ious, other_ious))
            print(f"  Better:    {100 * num_better / len(units):.2f}%")

        # Explanation metrics on the units where the beam searches differ
        if len(different_units) > 0:
            different_ious = [search_results[unit]["iou"] for unit in different_units]
            detection_accuracies = [search_results[unit]["detection_accuracy"] for unit in different_units]
            activation_coverages = [search_results[unit]["activation_coverage"] for unit in different_units]
            print(f"Found {len(different_units)} units with a different IoU:")
            print(f"\tIoU:     {np.mean(different_ious):.3f} ± {np.std(different_ious):.3f}")
            print(f"\tDetAcc:  {np.mean(detection_accuracies):.3f} ± {np.std(detection_accuracies):.3f}")
            print(f"\tActCov:  {np.mean(activation_coverages):.3f} ± {np.std(activation_coverages):.3f}")

        if search_name in beam_searches:
            visited = [search_results[unit]["visited"] for unit in units]
            estimated = [search_results[unit]["estimated"] for unit in units]
            print(f"\tVisited:   {np.mean(visited):.0f} ± {np.std(visited):.0f}, median {np.median(visited):.0f}")
            print(f"\tEstimated: {np.mean(estimated):.0f} ± {np.std(estimated):.0f}")
        if len(times) == 0:
            print("\ts/u:       -\n")
        elif len(times) < len(units):
            print(f"\ts/u:       {np.mean(times):.0f} ± {np.std(times):.0f}, median {np.median(times):.0f}, "
                  f"on the {len(times)} units with a time\n")
        else:
            print(f"\ts/u:       {np.mean(times):.0f} ± {np.std(times):.0f}, median {np.median(times):.0f}\n")


if __name__ == "__main__":
    absl.flags.mark_flags_as_required(["model", "dataset", "layer", "length", "beam_limit"])
    absl.app.run(main)
