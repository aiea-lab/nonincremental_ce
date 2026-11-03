"""Utilities to compute the dataset quantities and the NLI concepts. Adapted from
https://github.com/aiea-lab/optimal-compositional-explanations
"""

import copy
import os
import pickle

import torch
import numpy as np
import scipy.sparse as sparse
from scipy.spatial.distance import cdist

from src import constants, settings, snli


def get_dataset_quantities(masks: list, info_dir: str) -> tuple:
    """Computes (or loads) how many concepts annotate each pixel of the dataset.

    These quantities are used by the heuristic of the beam search to bound
    the IoU of a formula.

    Args:
        masks (list): sparse masks of shape (num_samples, H*W).
        info_dir (str): directory of the cache.

    Returns:
        tuple: boolean tensors of the common, unique, and uncoverable pixels.
    """
    file_path = f"{info_dir}/quantities.pkl"
    if os.path.exists(file_path):
        with open(file_path, "rb") as file:
            return pickle.load(file)

    # Number of concepts annotating each pixel
    stacked = sparse.vstack([mask.reshape(1, -1) for mask in masks]).tocsr()
    counts = np.asarray(stacked.sum(axis=0, dtype=np.int32))
    counts = torch.from_numpy(counts.reshape(masks[0].shape))
    common_elements = counts > 1
    unique_elements = counts == 1
    uncoverable_elements = counts == 0

    os.makedirs(info_dir, exist_ok=True)
    with open(file_path, "wb") as file:
        pickle.dump((common_elements, unique_elements, uncoverable_elements), file)
    return common_elements, unique_elements, uncoverable_elements


# ============================================================================
# NLI concepts
# ============================================================================



def get_nli_concepts(cfg: settings.Settings, data_loader: torch.utils.data.DataLoader) -> tuple:
    """Returns the concepts of the SNLI analysis dataset.

    Args:
        cfg (settings.Settings): settings of the run.
        data_loader (torch.utils.data.DataLoader): loader of the analysis dataset.

    Returns:
        tuple: masks, labels, and constraints of the concepts.
    """
    dataset = data_loader.dataset
    words, word_features, _ = snli.get_features(data_loader)
    concept_masks, concept_vocab = dataset.to_sentence(words, word_features)

    word_vectors_path = os.path.join(cfg.get_dataset_root(), "snli_1.0", "snli_1.0_dev.vec")
    concept_masks, concept_vocab, constraints = add_neighbor_concepts(
        concept_masks, concept_vocab, word_vectors_path=word_vectors_path
    )
    concept_labels = [concept_vocab["itos"][index] for index in range(len(concept_vocab["itos"]))]
    masks = [
        sparse.csr_matrix(concept_masks[:, [concept_index]])
        for concept_index in range(concept_masks.shape[1])
    ]
    return masks, concept_labels, constraints


def load_word_vectors(path: str) -> tuple:
    """Returns the word vectors of a .vec file (one word per line, followed by
    the values of its vector).

    Returns:
        tuple: vectors, word to index, and index to word.
    """
    vectors = []
    word_to_index = {}
    index_to_word = {}
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            word, *values = line.split(" ")
            assert word not in word_to_index
            index = len(word_to_index)
            word_to_index[word] = index
            index_to_word[index] = word
            vectors.append(np.array(list(map(float, values))))
    vectors = np.array(vectors)
    return vectors, word_to_index, index_to_word


def add_neighbor_concepts(concept_masks: np.ndarray, concept_vocab: dict, *, word_vectors_path: str) -> tuple:
    """Adds a neighbor concept for each word concept, which is the OR of the
    word and of its closest words. A neighbor concept cannot be combined with
    the concepts that generate it.

    Args:
        concept_masks (np.ndarray): boolean masks of shape
            (num_samples, num_concepts).
        concept_vocab (dict): vocabulary of the concepts.
        word_vectors_path (str): path of the word vectors.

    Returns:
        tuple: masks, vocabulary, and constraints of the concepts.
    """
    print(f"Loading the word vectors in {word_vectors_path}...")
    vectors, vector_word_to_index, vector_index_to_word = load_word_vectors(word_vectors_path)
    new_vocab = copy.deepcopy(concept_vocab)
    constraints = [[] for _ in range(len(concept_vocab["stoi"]))]
    num_words_with_vector = 0
    for concept_name, concept_index in concept_vocab["stoi"].items():
        part, kind, word = concept_name.split(":", maxsplit=2)
        if kind == "tag" or kind == "overlap":
            # Only word concepts have neighbors
            continue
        if word not in vector_word_to_index:
            continue
        num_words_with_vector += 1

        # Closest words. The closest one is the word itself
        word_vector = vectors[vector_word_to_index[word]][np.newaxis]
        distances = cdist(word_vector, vectors, metric="cosine")[0]
        nearest_indices = np.argsort(distances)[1 : constants.NLI_NEIGHBORHOOD_SIZE + 1]
        nearest_names = [f"{part}:{kind}:{vector_index_to_word[index]}" for index in nearest_indices]

        # Concepts that generate the neighbor concept
        generating_indices = [
            concept_vocab["stoi"][name] for name in nearest_names if name in concept_vocab["stoi"]
        ]
        generating_indices.append(concept_vocab["stoi"][concept_name])
        generating_indices = np.array(list(set(generating_indices)))
        if len(generating_indices) <= 1:
            continue

        neighbors_mask = np.logical_or.reduce(concept_masks[:, generating_indices], 1)
        for generating_index in generating_indices:
            generating_name = concept_vocab["itos"][generating_index]
            existing_neighbor_index = new_vocab["stoi"].get(f"neighbor:{generating_name}", None)
            # Skip the new concept if it is identical to the neighbor concept
            # of one of the words that generate it
            if existing_neighbor_index is not None and np.all(
                neighbors_mask == concept_masks[:, existing_neighbor_index]
            ):
                neighbors_mask = None
        if neighbors_mask is None or np.sum(neighbors_mask) == 0:
            continue

        neighbor_name = f"neighbor:{concept_name}"
        if neighbor_name not in new_vocab["stoi"]:
            neighbor_index = len(new_vocab["stoi"])
            new_vocab["stoi"][neighbor_name] = neighbor_index
            new_vocab["itos"][neighbor_index] = neighbor_name
            # Each generating concept cannot be combined with the neighbor concept
            for generating_index in generating_indices:
                constraints[generating_index].append(neighbor_index)
            constraints.append(generating_indices.tolist())
            assert len(constraints) == len(new_vocab["stoi"])
        concept_masks = np.concatenate([concept_masks, neighbors_mask[:, np.newaxis]], axis=1)
    print(f"Neighbor concepts computed for {num_words_with_vector} of {len(concept_vocab['stoi'])} concepts.")
    return concept_masks, new_vocab, constraints
