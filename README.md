# Beyond Incremental Beam Search for Compositional Explanations of Neurons
This repository provides the official implementation of the algorithm introduced in the paper **"Beyond Incremental Beam Search for Compositional Explanations of Neurons."** *Biagio La Rosa and Leilani H. Gilpin*. NeurIPS 2026.


## Contributions and their Implementation
Compositional explanations describe the alignment between neuron activations and concepts through logical formulas, obtained by solving a search problem over combinations of concepts.

Prior work relies on an incremental beam search, where the formulas of the beam are expanded by attaching one concept at a time to the current formula. Therefore, intermediate candidates that are not selected by the beam at early stages cannot be revisited later, even if they would lead to better explanations when combined differently.

**Our proposal**: move beyond the incremental formulation. Specifically, we introduce a non-incremental beam search that expands the search space by inserting concepts within the internal substructures of formulas and by recombining the best formulas explored at previous steps. We also introduce a functional-equivalence-aware beam search that keeps only one formula for each equivalence class of formulas that induce the same activation mask.

### Non-Incremental Beam Expansion


The non-incremental beam search expands the search space in two ways:
- **insertion**, which adds a concept within the internal substructures of a formula, and not only at its end;
- **recombination**, which generates candidates by combining the best formulas explored at the previous steps.


**Implementation**: the beam search is implemented in the `compositional/non_incremental_beam_search.py` file. The insertion is implemented in the `expand_formula_non_incremental` function, while the recombinations are implemented in the `combine_beam_formulas` and `combine_with_history` functions of the `utils/non_incremental_utils.py` file. 

---

### Functional-Equivalence-Aware Beam Search
Different logical formulas can induce the same activation mask. These formulas are functionally equivalent, and keeping more than one of them in the beam wastes beam positions.

The functional-equivalence-aware beam search identifies the candidates that are functionally equivalent to a formula of the beam and keeps only one formula for each equivalence class, thus increasing the diversity of the beam.

**Implementation**: the selection of the beam is implemented in the `select_beam_functional_aware` and `manage_equivalence` functions of the `compositional/non_incremental_beam_search.py` file, while the search of equivalent formulas is implemented in the `find_equivalent_formulas` function of the `utils/non_incremental_utils.py` file.

---

### Other Beam Variants
The repository also includes the incremental beam search in the `compositional/incremental_beam_search.py` file, and its variant guided by the M-MESH heuristic in the `compositional/mmesh_heuristic.py` file. The optimal explanations are computed with the [`compositional_explanations`](https://pypi.org/project/compositional-explanations/) package.

## Quick Start

### Setup the environment
Follow step 1 and 2 described in <a href="PAPER.md">PAPER.md</a> to set up the environment and download the required assets.

### Main CLI Flags
All the scripts share a set of user arguments that define the configuration for which explanations are generated.

Common flags are defined in `src/common_flags.py`.
The main ones are:
- `--dataset`: `broden` for the vision models, `snli` for the NLI models
- `--model`: `resnet18`, `alexnet`, `densenet161` for vision, `bowman`, `bert`, `llama` for NLI
- `--layer`: `layer4` for `resnet18`, `features` for `alexnet` and `densenet161`, `mlp` for the NLI models
- `--search`: `incremental`, `non_incremental`, `optimal`
- `--length`: explanation length
- `--beam_limit`: beam size (not used by the optimal search)
- `--units`: list of units to explain (default `none`, which means all the units)
- `--random_units`: number of random units to select (default `0`, which means all the units)
- `--device`: compute device (default `cuda`)

In addition, the user can set the following flags
- `--seed`: random seed (default `0`)
- `--regenerate`: whether to recompute the explanations already stored (default `False`)
- `--step_size`: number of concepts whose masks are extracted from the annotation dataset in a single step (default `80`). Larger values can speed up the computation but can be memory intensive.
- `--fast_impl`: whether to keep all the dataset segmentations in memory during the extraction of the masks (default `False`, can speed up computation but can be memory intensive)

Finally, the user can set the following flags to specify the output directories (if not using the default ones):
- `--root_datasets`: root directory where the datasets are stored (default `data/datasets`)
- `--root_models`: root directory where the models are stored (default `data/model`)
- `--root_results`: root directory where the results are stored (default `data/results`)
- `--root_segmentations`: root directory where the concept masks are stored (default `data/cache/segmentations`)
- `--root_activations`: root directory where the activations are stored (default `data/cache/activations`)

### Compute Explanations
To compute explanations, use the `main.py` script with the common flags set appropriately.

```bash
python3 main.py --dataset=<DATASET> --model=<MODEL> --layer=<LAYER> --search=<SEARCH> --length=<LENGTH> --beam_limit=<BEAM_SIZE>
```

For example, to explain unit 1 of BERT with the non-incremental beam search:

```bash
python3 main.py --dataset=snli --model=bert --layer=mlp --search=non_incremental --length=5 --beam_limit=10 --units=1
```

Explanations are saved in `data/results/`. Specifically, the explanations of each search are saved in separate subdirectories:
- `data/results/<DATASET>/<MODEL>/<LAYER>/incremental/`
- `data/results/<DATASET>/<MODEL>/<LAYER>/non_incremental/`
- `data/results/<DATASET>/<MODEL>/<LAYER>/optimal/`


### Generate Qualitative Difference Visualizations
Generate qualitative difference visualizations between the incremental and the non-incremental beam searches on the units where they find a different IoU:

```bash
python3 visualize.py --dataset=<DATASET> --model=<MODEL> --layer=<LAYER> --length=<LENGTH> --beam_limit=<BEAM_SIZE>
```

- Visualizations are saved under `data/figures/<DATASET>/<MODEL>/<LAYER>/`.
- For the vision models, `unit_<UNIT>.png` shows, for each explanation, the samples where both the unit and the explanation are active.
- For the NLI models, `unit_<UNIT>.txt` lists, for each explanation, the top-activating premise and hypothesis pairs where the explanation holds, and whether the explanation of the other search also holds on them.


## Citation

If you use this repository, please cite:

```bibtex
@inproceedings{
LaRosa2026,
title={Beyond Incremental Beam Search for Compositional Explanations of Neurons},
author={Biagio {La Rosa} and Leilani H. Gilpin},
booktitle={The Fortieth Annual Conference on Neural Information Processing Systems},
year={2026},
url={https://openreview.net/forum?id=QYZFm5iY4H}
}
```
