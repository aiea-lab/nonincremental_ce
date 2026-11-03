
# Reproducing Paper results

This document provides instructions for reproducing the results in the paper.

## Setup Repo

### Step 1: Create environment.

The suggested option for creating the environment is to use Docker.

Build the image from the `Dockerfile` of the repository, which is based on `nvcr.io/nvidia/pytorch:23.01-py3` (<a href="https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-23-01.html">here</a>) from nvidia NGC and installs the dependencies listed in `requirements.txt`:

```bash
docker build -t non_incremental .
```

Then, run the container:

```bash
docker run --gpus all -it --rm -v $(pwd):/workspace --ipc=host non_incremental
```

As an alternative to Docker, you can also create a Python virtual environment and install the dependencies to replicate the image. In this case, some numbers could slightly differ due to activation differences when running on different setups.

### Step 2: Download Required Assets

#### Models
From the repository root:

```bash
bash scripts/download_models.sh
```

This downloads all probed models: ResNet-18, AlexNet, and DenseNet-161 trained on Places365, and the BiLSTM, BERT, and LLaMA models trained on SNLI. It also adds the vocabulary of the BiLSTM checkpoint to the BERT and LLaMA checkpoints.

#### Datasets
From the repository root:

```bash
bash scripts/download_datasets.sh
```

This will populate:

- `data/datasets/broden/` with `broden1_224` and `broden1_227`
- `data/datasets/snli_1.0/` with the dev split of SNLI and the preprocessed features and word vectors of [Mu and Andreas (2020)](https://github.com/jayelm/compexp/tree/master/nli)

## Step 3: Run scripts

### Numerical Results
These are the main commands needed to reproduce the paper's core numerical results.

For each of these, first run `main.py` with the appropriate flags, then run `compare_methods.py` with the exact same flags to get summary statistics.

**Note**: The first time you run `main.py` on a model, it will generate and save the activations, the concept masks for all the concepts in the dataset, and the quantities used by the heuristic. This can take a long time, especially for Broden. However, once they are cached, subsequent runs on the same dataset will load these precomputed values and be significantly faster.

### Table 1, Table 3, and Table 4

To run experiments for Tables 1, 3, and 4 on the natural language inference task:
```bash
python3 main.py --dataset=snli --model=<MODEL> --layer=mlp --search=<ALGORITHM> --length=5 --beam_limit=10
```
Replace \<MODEL> and \<ALGORITHM> with the appropriate values:

- \<MODEL>: use `bowman` for the BiLSTM, `bert` for BERT, and `llama` for LLaMA.

- \<ALGORITHM>: use `incremental` or `non_incremental` to run the incremental beam search or our non-incremental beam search, respectively.

Then run `compare_methods.py` with the same flags to get summary statistics:
```bash
python3 compare_methods.py --dataset=snli --model=<MODEL> --layer=mlp --length=5 --beam_limit=10
```
The units of the BiLSTM with too few activations are skipped, so its statistics are computed on the units that can be explained.

### Table 2, Table 5, and Table 6

To run experiments for Tables 2, 5, and 6 on image classification:
```bash
python3 main.py --dataset=broden --model=<MODEL> --layer=<LAYER> --search=<ALGORITHM> --length=3 --beam_limit=5
```
Replace \<MODEL>, \<LAYER>, and \<ALGORITHM> with the appropriate values:

- \<MODEL> and \<LAYER>: use `resnet18` and `layer4` for ResNet-18, `alexnet` and `features` for AlexNet, and `densenet161` and `features` for DenseNet-161.

- \<ALGORITHM>: use `incremental`, `non_incremental`, or `optimal` to run the incremental beam search, our non-incremental beam search, or the optimal algorithm, respectively. The optimal algorithm ignores `--beam_limit`.

Then run `compare_methods.py` with the same flags to get summary statistics:
```bash
python3 compare_methods.py --dataset=broden --model=<MODEL> --layer=<LAYER> --length=3 --beam_limit=5
```
The optimal algorithm can take more than one hour per unit. When it is run only on some units, `compare_methods.py` computes the statistics of Table 2 only on these units. To get the statistics ignoring the optimal algorithm, use `--include_optimal=false`.

### Table 7

To run the experiments for Table 7, run the incremental beam search with the M-MESH heuristic:
```bash
python3 mmesh.py --dataset=broden --model=<MODEL> --layer=<LAYER> --length=3 --beam_limit=5
```
with \<MODEL> and \<LAYER> as in Table 2. Then compare it with the incremental beam search of Table 2:
```bash
python3 compare_methods.py --dataset=broden --model=<MODEL> --layer=<LAYER> --length=3 --beam_limit=5 --compared_searches=incremental,incremental_mmesh --include_optimal=false
```

### Table 8

To run the experiments for Table 8:
```bash
python3 ablation.py --dataset=snli --model=bert --layer=mlp --length=5 --beam_limit=10 --ablation=<ABLATION>
```
Replace \<ABLATION> with `no_recombinations`, `no_functional_equivalence`, or `no_non_incremental_expansion`. Then compare each ablation with the full version of Table 1:
```bash
python3 compare_methods.py --dataset=snli --model=bert --layer=mlp --length=5 --beam_limit=10 --compared_searches=<ABLATION>,non_incremental --include_optimal=false
```

### Table 9

To run the experiment for Table 9:
```bash
python3 main.py --dataset=snli --model=bert --layer=mlp --search=<ALGORITHM> --length=5 --beam_limit=50 --random_units=50
```
with \<ALGORITHM> equal to `incremental` and `non_incremental`. Then run `compare_methods.py` with the same flags, except `--search`.

### Figures 3, 4, 5, and 6

To run the experiments for Figures 3 and 4, run `main.py` as in Table 9 with `--beam_limit` equal to `10`, `25`, and `50`. For Figures 5 and 6, run it with `--beam_limit=10` and `--length` equal to `5`, `10`, and `20`. Then generate the runtimes with:
```bash
python3 sensitivity.py --dataset=snli --model=bert --layer=mlp --sweep_beam_limits=10,25,50 --sweep_lengths=5
```
for Figures 3 and 4, and with `--sweep_beam_limits=10 --sweep_lengths=5,10,20` for Figures 5 and 6. The script computes the runtimes on the units explained in every configuration.

### Images

To generate the images included in the paper, first run `main.py` as in Tables 1 and 2. Then generate the images with:
```bash
python3 visualize.py --dataset=<DATASET> --model=<MODEL> --layer=<LAYER> --length=<LENGTH> --beam_limit=<BEAM_SIZE>
```
using the flags of Table 1 for the NLI models and of Table 2 for the vision models. The script generates one image for each unit where the two beam searches find a different IoU. Use `--units` to generate only the units shown in the paper:

| Model | Figures | `--units` |
|---|---|---|
| BiLSTM (`bowman`) | 1, 13-18 | `377,0,15,16,17,19,22` |
| BERT (`bert`) | 19-24 | `0,1,2,3,4,5` |
| LLaMA (`llama`) | 25-30 | `0,1,2,3,4,5` |
| ResNet-18 (`resnet18`) | 7, 8 | `30,39,45,56,70,87` |
| AlexNet (`alexnet`) | 2, 9, 10 | `91,2,22,29,31,37,50` |
| DenseNet-161 (`densenet161`) | 11, 12 | `1,4,5,7,8,22` |

## Reproducibility Notes

- Set `--seed` (default `0`) for deterministic behavior where possible. The same seed also selects the same units with `--random_units`.
- The numbers of the paper were computed in the Docker image of Step 1. Other environments and GPUs can produce slightly different activations, and therefore slightly different explanations.
- The Broden runs need about 17 GB of RAM. Consider running one Broden job at a time and tuning `--step_size` or `--fast_impl`.

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
