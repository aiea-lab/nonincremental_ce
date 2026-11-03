"""Data loaders of the Broden (vision) and SNLI (NLI) datasets."""

import os
from typing import Callable, Optional

import torch
import torchvision

from src import constants, segmentations, settings, snli
from utils import common_utils 

def get_broden_data(cfg: settings.Settings, *, resolution: int) -> torch.utils.data.DataLoader:
    """Returns the data loader of the Broden dataset at the given resolution."""
    dataset = segmentations.BrodenDataset(
        cfg.dir_datasets,
        resolution=resolution,
        broden_version=1,
        transform_image=torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((resolution, resolution)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    cfg.get_image_mean(), cfg.get_image_stdev()
                ),
            ]
        ),
    )
    generator = common_utils.get_generator(constants.DATA_LOADER_SEED)
    segmentation_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        worker_init_fn=common_utils.seed_worker,
        generator=generator,
    )
    return segmentation_loader

def get_data_loader(cfg: settings.Settings, vocab: Optional[dict] = None) -> torch.utils.data.DataLoader:
    """Returns the data loader of the dataset of the run.

    Args:
        cfg (settings.Settings): settings of the run.
        vocab (dict): vocabulary of the NLI model.

    Returns:
        torch.utils.data.DataLoader: the data loader.
    """
    if cfg.get_dataset_name() == 'broden':
        return get_broden_data(cfg, resolution=cfg.get_img_size())
    elif cfg.get_dataset_name() == 'snli':
        return get_nli_data(cfg, vocab)
    else:
        raise NotImplementedError(f"Dataset {cfg.get_dataset_name()} not supported yet")

def decompose_batch_nli(batch: tuple) -> tuple:
    """Splits a batch of SNLI sentences into premises and hypotheses.

    Returns:
        tuple: premises, premise lengths, hypotheses, hypothesis lengths, and
            sentence indices.
    """
    sentences, _, _, sentence_lengths, sentence_indices = batch
    with torch.no_grad():
        sentence_pairs = snli.pairs(sentences.squeeze(2))
        length_pairs = snli.pairs(sentence_lengths)

        premises = sentence_pairs[:, :, 0]
        premise_lengths = length_pairs[:, 0]
        hypotheses = sentence_pairs[:, :, 1]
        hypothesis_lengths = length_pairs[:, 1]

    return premises, premise_lengths, hypotheses, hypothesis_lengths, sentence_indices


def get_nli_dataset(cfg: settings.Settings, vocab: dict) -> snli.AnalysisDataset:
    """Returns the analysis dataset of the SNLI dev set."""
    snli_dir = os.path.join(cfg.get_dataset_root(), "snli_1.0")
    with open(os.path.join(snli_dir, "snli_1.0_dev.feats"), "r") as file:
        lines = file.readlines()
    return snli.AnalysisDataset(lines, vocab, path=snli_dir + "/", split="dev")


def get_nli_data(cfg: settings.Settings, vocab: dict) -> torch.utils.data.DataLoader:
    """Returns the data loader of the analysis dataset of the SNLI dev set."""
    dataset = get_nli_dataset(cfg, vocab)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        batch_size=constants.NLI_BATCH_SIZE,
        collate_fn=lambda batch: snli.pad_collate(batch, sort=False),
    )
    return data_loader


def get_data_iter_fn(dataset_name: str) -> Callable:
    """Returns the function that extracts the input of the model from a batch.

    Args:
        dataset_name (str): name of the dataset, broden or snli.

    Returns:
        callable: function that takes a batch and returns the input of the model.
    """
    if dataset_name == 'broden':
        return lambda x: x[0]
    elif dataset_name == 'snli':
        return lambda x: decompose_batch_nli(x)
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not supported yet")
