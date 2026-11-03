"""Utilities to set the seeds for reproducibility and to store the
explanations computed for each unit.
"""

import glob
import os
import pickle
import random
from typing import Optional

import torch
import numpy as np


def set_seed(seed: int):
    """Sets the seed of the random number generators and makes torch
    deterministic.

    Args:
        seed (int): seed to use.
    """
    # Deterministic cuBLAS operations require a fixed workspace size
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    torch.use_deterministic_algorithms(True)


def get_generator(seed: int) -> torch.Generator:
    """Returns a torch generator with the given seed.

    Args:
        seed (int): seed of the generator.

    Returns:
        torch.Generator: the seeded generator.
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


# reference: https://pytorch.org/docs/stable/notes/randomness.html
def seed_worker(worker_id: int):
    """Sets the seed of a worker of the data loader.

    Args:
        worker_id (int): id of the worker.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_search_results_dir(results_dir: str, *, search_name: str, beam_limit: Optional[int], length: int) -> str:
    """Returns the directory storing the explanations of a search.

    Args:
        results_dir (str): directory of the explanations of the layer.
        search_name (str): name of the search.
        beam_limit (int): beam size, or None.
        length (int): maximum length of the explanations.

    Returns:
        str: directory of the explanations of the search.
    """
    if beam_limit is None:
        # The optimal search has no beam
        search_dir = os.path.join(results_dir, search_name)
    else:
        search_dir = os.path.join(results_dir, search_name, f"beam_{beam_limit}")
    return os.path.join(search_dir, f"length_{length}")


def get_unit_results_path(
    results_dir: str,
    *,
    search_name: str,
    beam_limit: Optional[int],
    length: int,
    unit: int,
    threshold: float,
) -> str:
    """Returns the path of the file storing the explanation of a unit.

    The threshold is part of the path.

    Args:
        results_dir (str): directory of the explanations of the layer.
        search_name (str): name of the search.
        beam_limit (int): beam size, or None.
        length (int): maximum length of the explanations.
        unit (int): index of the unit.
        threshold (float): activation threshold of the unit.

    Returns:
        str: path of the results file.
    """
    search_dir = get_search_results_dir(
        results_dir, search_name=search_name, beam_limit=beam_limit, length=length
    )
    return os.path.join(search_dir, f"unit_{unit}", f"threshold_{threshold}.pickle")


def save_unit_results(path: str, unit_results: dict):
    """Stores the explanation of a unit.

    Args:
        path (str): path of the results file.
        unit_results (dict): explanation of the unit and statistics of the search.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as results_file:
        pickle.dump(unit_results, results_file)


def load_unit_results(path: str) -> dict:
    """Loads the explanation of a unit stored with save_unit_results.

    Args:
        path (str): path of the results file.

    Returns:
        dict: explanation of the unit and statistics of the search.
    """
    with open(path, "rb") as results_file:
        return pickle.load(results_file)


def load_search_results(results_dir: str, *, search_name: str, beam_limit: Optional[int], length: int) -> dict:
    """Loads the explanations of all the units explained by a search.

    Args:
        results_dir (str): directory of the explanations of the layer.
        search_name (str): name of the search.
        beam_limit (int): beam size, or None.
        length (int): maximum length of the explanations.

    Returns:
        dict: explanation of each unit. Empty if the search has no results.
    """
    search_dir = get_search_results_dir(
        results_dir, search_name=search_name, beam_limit=beam_limit, length=length
    )
    results = {}
    for unit_dir in glob.glob(os.path.join(search_dir, "unit_*")):
        unit = int(os.path.basename(unit_dir)[len("unit_"):])
        unit_files = glob.glob(os.path.join(unit_dir, "threshold_*.pickle"))
        if len(unit_files) > 1:
            raise ValueError(
                f"Unit {unit} has results for more than one threshold in {unit_dir}: "
                "keep only the ones computed on the current activations."
            )
        results[unit] = load_unit_results(unit_files[0])
    return results
