"""
Models trained on SNLI. The BiLSTM model is taken from the CoEx repository
(https://github.com/jayelm/compexp) and the BERT and LLaMA classifiers are
adapted from the CCE_NLI repository
(https://github.com/rishikasrinivas/CCE_NLI/tree/cofi_merging).
"""

from typing import Optional

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class BowmanEntailmentClassifier(nn.Module):
    """
    The RNN-based entailment model of Bowman et al 2017
    """

    def __init__(self, encoder: "TextEncoder"):
        super().__init__()

        self.encoder = encoder
        self.encoder_dim = encoder.output_dim
        self.mlp_input_dim = self.encoder_dim * 4
        self.dropout = nn.Dropout(0.1)
        self.bn = nn.BatchNorm1d(self.mlp_input_dim)

        self.mlp = nn.Sequential(
            nn.Linear(self.mlp_input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),  # Mimic classifier MLP keep rate of 94%
            nn.Linear(1024, 3),
        )

        self.output_dim = 3

    def forward(self, s1: torch.Tensor, s1len: torch.Tensor, s2: torch.Tensor, s2len: torch.Tensor) -> torch.Tensor:
        s1enc = self.encoder(s1, s1len)
        s2enc = self.encoder(s2, s2len)

        diffs = s1enc - s2enc
        prods = s1enc * s2enc

        mlp_input = torch.cat([s1enc, s2enc, diffs, prods], 1)

        mlp_input = self.bn(mlp_input)
        mlp_input = self.dropout(mlp_input)

        preds = self.mlp(mlp_input)

        return preds

    def get_final_reprs(
        self, s1: torch.Tensor, s1len: torch.Tensor, s2: torch.Tensor, s2len: torch.Tensor
    ) -> torch.Tensor:
        s1enc = self.encoder(s1, s1len)
        s2enc = self.encoder(s2, s2len)

        diffs = s1enc - s2enc
        prods = s1enc * s2enc

        mlp_input = torch.cat([s1enc, s2enc, diffs, prods], 1)

        mlp_input = self.bn(mlp_input)
        mlp_input = self.dropout(mlp_input)

        rep = self.mlp[:-1](mlp_input)
        return rep

    def forward_from_final(self, rep: torch.Tensor) -> torch.Tensor:
        preds = self.mlp[-1:](rep)
        return preds


class TextEncoder(nn.Module):
    def __init__(
        self, vocab_size: int, embedding_dim: int = 300, hidden_dim: int = 512, bidirectional: bool = False
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.bidirectional = bidirectional
        self.emb = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=1)
        self.rnn = nn.LSTM(
            self.embedding_dim, self.hidden_dim, bidirectional=bidirectional
        )
        self.output_dim = self.hidden_dim

    def forward(self, s: torch.Tensor, slen: torch.Tensor) -> torch.Tensor:
        semb = self.emb(s)
        spk = pack_padded_sequence(semb, slen.cpu(), enforce_sorted=False)
        _, (states, _) = self.rnn(spk)

        return states[-1]


class LLAMAEntailmentClassifier(nn.Module):
    """LLaMA encoder with the classifier of the BiLSTM model."""

    # Adapted from LLAMAEntailmentClassifier of CCE_NLI
    def __init__(self, encoder_name: str = "knowledgator/Llama-encoder-1.0B"):
        """
        Args:
            encoder_name (str): name of the pretrained encoder.
        """
        super().__init__()
        from llm2vec.models import LlamaBiModel
        from transformers import AutoTokenizer

        self.encoder_name = encoder_name
        self.model = LlamaBiModel.from_pretrained(encoder_name)
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_name)
        self.tokenizer.model_max_length = 512
        # Add a padding token, as during training
        if "pad_token" not in self.tokenizer.special_tokens_map:
            num_new_tokens = self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            if num_new_tokens > 0:
                self.model.resize_token_embeddings(len(self.tokenizer))

        self.encoder_dim = self.model.config.hidden_size
        self.mlp_input_dim = self.encoder_dim * 4
        self.dropout = nn.Dropout(0.1)
        self.bn = nn.BatchNorm1d(self.mlp_input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.mlp_input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 3),
        )
        self.output_dim = 3

    # Adapted from LLAMAEntailmentClassifier of CCE_NLI
    def forward(
        self, s1_tokens: dict, s1len: Optional[torch.Tensor], s2_tokens: dict, s2len: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Returns the logits of the premise-hypothesis pairs.

        Args:
            s1_tokens (dict): tokenized premises.
            s1len: not used.
            s2_tokens (dict): tokenized hypotheses.
            s2len: not used.

        Returns:
            torch.Tensor: logits of shape (B, 3).
        """
        rep = self.get_final_reprs(s1_tokens, s1len, s2_tokens, s2len)
        return self.forward_from_final(rep)

    # Adapted from LLAMAEntailmentClassifier of CCE_NLI
    def get_final_reprs(
        self, s1_tokens: dict, s1len: Optional[torch.Tensor], s2_tokens: dict, s2len: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Returns the activations of the hidden layer of the MLP.

        Args:
            s1_tokens (dict): tokenized premises.
            s1len: not used.
            s2_tokens (dict): tokenized hypotheses.
            s2len: not used.

        Returns:
            torch.Tensor: activations of shape (B, 1024).
        """
        s1enc = self.encode_sentence(s1_tokens)
        s2enc = self.encode_sentence(s2_tokens)

        diffs = s1enc - s2enc
        prods = s1enc * s2enc

        mlp_input = torch.cat([s1enc, s2enc, diffs, prods], 1)
        mlp_input = self.bn(mlp_input)
        mlp_input = self.dropout(mlp_input)
        rep = self.mlp[:-1](mlp_input)
        return rep

    def forward_from_final(self, rep: torch.Tensor) -> torch.Tensor:
        preds = self.mlp[-1:](rep)
        return preds

    # Adapted from LLAMAEntailmentClassifier of CCE_NLI
    def encode_sentence(self, tokens: dict) -> torch.Tensor:
        """Returns the mean of the hidden states of the tokens, excluding the
        padding."""
        hidden_states = self.model(**tokens).last_hidden_state
        token_mask = tokens["attention_mask"].unsqueeze(-1).float()
        num_tokens = token_mask.sum(dim=1).clamp(min=1e-9)
        return (hidden_states * token_mask).sum(dim=1) / num_tokens


class BertEntailmentClassifier(nn.Module):
    """BERT encoder with the classifier of the BiLSTM model."""

    # Adapted from BertEntailmentClassifier of CCE_NLI
    def __init__(self, encoder_name: str = "bert-base-uncased"):
        """
        Args:
            encoder_name (str): name of the pretrained encoder.
        """
        super().__init__()
        from transformers import AutoModel, AutoTokenizer

        self.encoder_name = encoder_name
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_name)
        self.bert = AutoModel.from_pretrained(encoder_name)

        self.encoder_dim = self.bert.config.hidden_size
        self.mlp_input_dim = self.encoder_dim * 4
        self.dropout = nn.Dropout(0.1)
        self.bn = nn.BatchNorm1d(self.mlp_input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.mlp_input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 3),
        )
        self.output_dim = 3

    # Adapted from BertEntailmentClassifier of CCE_NLI
    def forward(
        self, s1_tokens: dict, s1len: Optional[torch.Tensor], s2_tokens: dict, s2len: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Returns the logits of the premise-hypothesis pairs.

        Args:
            s1_tokens (dict): tokenized premises.
            s1len: not used.
            s2_tokens (dict): tokenized hypotheses.
            s2len: not used.

        Returns:
            torch.Tensor: logits of shape (B, 3).
        """
        rep = self.get_final_reprs(s1_tokens, s1len, s2_tokens, s2len)
        return self.forward_from_final(rep)

    # Adapted from BertEntailmentClassifier of CCE_NLI
    def get_final_reprs(
        self, s1_tokens: dict, s1len: Optional[torch.Tensor], s2_tokens: dict, s2len: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Returns the activations of the hidden layer of the MLP.

        Args:
            s1_tokens (dict): tokenized premises.
            s1len: not used.
            s2_tokens (dict): tokenized hypotheses.
            s2len: not used.

        Returns:
            torch.Tensor: activations of shape (B, 1024).
        """
        s1enc = self.encode_sentence(s1_tokens)
        s2enc = self.encode_sentence(s2_tokens)

        diffs = s1enc - s2enc
        prods = s1enc * s2enc

        mlp_input = torch.cat([s1enc, s2enc, diffs, prods], 1)
        mlp_input = self.bn(mlp_input)
        mlp_input = self.dropout(mlp_input)
        rep = self.mlp[:-1](mlp_input)
        return rep

    def forward_from_final(self, rep: torch.Tensor) -> torch.Tensor:
        preds = self.mlp[-1:](rep)
        return preds

    def encode_sentence(self, tokens: dict) -> torch.Tensor:
        outputs = self.bert(**tokens)
        return outputs.last_hidden_state[:, 0, :]  # embedding of the [CLS] token
