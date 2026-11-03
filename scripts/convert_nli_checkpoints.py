"""Script to add the vocabulary of the BiLSTM checkpoint to the BERT and LLaMA
checkpoints."""

import os

import torch

MODEL_DIR = os.path.join("data", "model", "other")
VOCABULARY_CHECKPOINT = "bowman.pth"
CHECKPOINTS_TO_CONVERT = ["bert_dense.pth", "llama_dense.pth"]


def main():
    vocabulary_checkpoint = torch.load(os.path.join(MODEL_DIR, VOCABULARY_CHECKPOINT), map_location="cpu")
    vocabulary = {"itos": vocabulary_checkpoint["itos"], "stoi": vocabulary_checkpoint["stoi"]}

    for file_name in CHECKPOINTS_TO_CONVERT:
        path = os.path.join(MODEL_DIR, file_name)
        checkpoint = torch.load(path, map_location="cpu")
        if "itos" in checkpoint:
            print(f"{file_name}: the vocabulary is already stored")
            continue
        checkpoint["itos"] = vocabulary["itos"]
        checkpoint["stoi"] = vocabulary["stoi"]
        # Save to a temporary file first, so an interrupted save does not
        # corrupt the checkpoint
        temporary_path = path + ".tmp"
        torch.save(checkpoint, temporary_path)
        os.replace(temporary_path, path)
        print(f"{file_name}: vocabulary added with {len(vocabulary['itos'])} words")


if __name__ == "__main__":
    main()
