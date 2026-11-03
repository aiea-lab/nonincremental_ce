"""
SNLI analysis dataset. Adapted from the CoEx repository
(https://github.com/jayelm/compexp).
"""


import os
from typing import Optional

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

PAD_IDX = 1
UNK_TOKEN = "UNK"
LABEL_STOI = {"entailment": 0, "neutral": 1, "contradiction": 2}
LABEL_ITOS = {v: k for k, v in LABEL_STOI.items()}
NO_GOLD_LABEL = -1
LABEL_ITOS[NO_GOLD_LABEL] = "-"
N_SENTENCE_FEATS = 2000  # how many of the most common sentence lemmas to keep

def pairs(x: torch.Tensor) -> torch.Tensor:
    """
    (max_len, batch_size, *feats)
    -> (max_len, batch_size / 2, 2, *feats)
    """
    if x.ndim == 1:
        return x.unsqueeze(1).view(-1, 2)
    else:
        return x.unsqueeze(2).view(x.shape[0], -1, 2, *x.shape[2:])

# Adapted from extract_features of CoEx
def get_features(data_loader: torch.utils.data.DataLoader) -> tuple:
    """Returns the words and the features of the premise-hypothesis pairs.

    Args:
        data_loader (torch.utils.data.DataLoader): data loader of the analysis
            dataset.

    Returns:
        tuple: word ids, features, and dataset indices of the pairs.
    """
    all_srcs = []
    all_feats = []
    all_multifeats = []
    all_idxs = []
    for src, src_feats, src_multifeats, src_lengths, idx in data_loader:
        # Memory bank - hidden states for each step
        with torch.no_grad():
            # Combine q/h pairs
            src_one = src.squeeze(2)
            src_one_comb = pairs(src_one)

        # Pack the sequence
        all_srcs.extend(list(np.transpose(src_one_comb.cpu().numpy(), (1, 2, 0))))
        all_feats.extend(
            list(np.transpose(pairs(src_feats).cpu().numpy(), (1, 2, 0, 3)))
        )
        all_multifeats.extend(
            list(np.transpose(pairs(src_multifeats).cpu().numpy(), (1, 2, 0, 3)))
        )
        all_idxs.extend(list(pairs(idx).cpu().numpy()))

    all_feats = {"onehot": all_feats, "multi": all_multifeats}

    return all_srcs,  all_feats, all_idxs

def pad_collate(batch: list, sort: bool = True) -> tuple:
    src, src_feats, src_multifeats, src_len, idx = zip(*batch)
    idx = torch.tensor(idx)
    src_len = torch.tensor(src_len)
    src_pad = pad_sequence(src, padding_value=PAD_IDX)
    # NOTE: part of speeches are padded with 0 - we don't actually care here
    src_feats_pad = pad_sequence(src_feats, padding_value=-1)
    src_multifeats_pad = pad_sequence(src_multifeats, padding_value=-1)

    if sort:
        src_len_srt, srt_idx = torch.sort(src_len, descending=True)
        src_pad_srt = src_pad[:, srt_idx]
        src_feats_pad_srt = src_feats_pad[:, srt_idx]
        src_multifeats_pad_srt = src_multifeats_pad[:, srt_idx]
        idx_srt = idx[srt_idx]
        return (
            src_pad_srt,
            src_feats_pad_srt,
            src_multifeats_pad_srt,
            src_len_srt,
            idx_srt,
        )
    return src_pad, src_feats_pad, src_multifeats_pad, src_len, idx


class AnalysisDataset:
    # Adapted from AnalysisDataset of CoEx
    def __init__(self, text_raw: list, fields: dict, *, path: str, split: str):
        """
        Initialize an analysis dataset from the given text file and vocab
        (`fields`) file which gives word-level annotations.

        Closed categories are categories whose # feats is constant w.r.t. dataset size.

        Open categories are those whose # feats grows roughly linear in the dataset
        size (e.g. lemmas). We keep these as separate categories and do not
        automatically search for these.

        Args:
            text_raw (list): lines of the .feats file.
            fields (dict): vocabulary of the model.
            path (str): directory of the SNLI text files.
            split (str): SNLI split.
        """
        self.words = []
        self.fs = []
        self.mfs = []
        self.lengths = []
        self.fields = fields

        self.stoi = self.fields["stoi"]
        self.itos = self.fields["itos"]

        self.pos_stoi = {}

        first_line = text_raw[0]
        self.byte_fmt = isinstance(first_line, bytes)
        if self.byte_fmt:
            first_line = first_line.decode("utf-8")
        # FIXME: closed feats should be part of the shared vocab (pos:)...etc.
        (
            self.cs,
            self.cnames,
            self.ctypes,
            self.ocnames,
            self.ccnames,
            self.mcnames,
        ) = self.compute_cat_names(first_line)

        self.text = text_raw[1:]
        self.targets = self.load_labels(path=path, split=split)
        self.target_itos = LABEL_ITOS

        # Mapping legend:
        # feats (short: f) are the individual values
        # cats (short: c) are the categories (e.g. POS, Lemma, Synset)
        # Mappings - feat names to individual features
        self.cnames2fis = {cn: set() for cn in self.cnames}
        self.parse_docs()
        self.fis2cnames = {}
        for cname, fis in self.cnames2fis.items():
            for fi in fis:
                self.fis2cnames[fi] = cname

        self.citos = dict(zip(self.cs, self.cnames))
        self.cstoi = dict(zip(self.cnames, self.cs))

        self.cis2fis = {
            self.cstoi[cname]: fis for cname, fis in self.cnames2fis.items()
        }
        self.fis2cis = {}
        for ci, fis in self.cis2fis.items():
            for fi in fis:
                self.fis2cis[fi] = ci

        self.n_cs = len(self.cnames)
        self.n_fs = len(self.fstoi)
        self.cfis = []
        self.ofis = []
        self.mfis = []
        for cname, fis in self.cnames2fis.items():
            ctype = self.ctypes[cname]
            if ctype == "open":
                self.ofis.extend(fis)
            elif ctype == "multi":
                self.mfis.extend(fis)
            else:
                self.cfis.extend(fis)

        for ws, fs, mfs in self.docs:
            self.words.append(np.array(ws))
            self.fs.append(np.array(fs))
            self.mfs.append(mfs)
            self.lengths.append(len(ws))

    # Adapted from the SNLI dataset of CoEx
    def load_labels(self, *, path: str, split: str) -> list:
        """Returns the gold label of each pair of the split.

        Args:
            path (str): directory of the SNLI text files.
            split (str): SNLI split.

        Returns:
            list: label of each pair.
        """
        self.path = path
        self.split = split
        assert self.split in {"train", "dev", "test"}
        self.text_path = os.path.join(self.path, f"snli_1.0_{self.split}.txt")

        labels = []
        with open(self.text_path, "r") as f:
            for i, line in enumerate(f):

                if i == 0:  # Header
                    continue

                label, _, _, _, _, s1, s2, *_ = line.strip().split("\t")
                # The pairs without a gold label, marked with "-" in the file, get NO_GOLD_LABEL
                if label not in LABEL_STOI:
                    label_i = NO_GOLD_LABEL
                else:
                    label_i = LABEL_STOI[label]
                labels.append(label_i)
        return labels

                
    def parse_docs(self):
        """
        Generator to parse documents, tokens, and feats
        """
        self.docs = []
        self.fstoi = {UNK_TOKEN: 0} # feature string to id
        self.fitos = {0: UNK_TOKEN} # feature id to string
        self.idx2multi = {}
        self.multi2idx = {}
        for line in self.text:
            line = line.strip()
            if self.byte_fmt:
                line = line.decode("utf-8")
            doc_words = []
            doc_feats = []
            doc_multifeats = []
            for tok in line.split(" "):
                word, *feats = tok.split("|")
                word_n = self.stoi.get(word.lower(), self.stoi["UNK"])
                feats = dict(zip(self.cnames, feats))
                feats_p = []
                multifeats_p = []
                for fn, f in feats.items():
                    if self.is_multi(fn):
                        fs = f.split(";")
                        fs_n = []
                        for f in fs:
                            # First assign global feature id
                            f = f"{fn}:{f}"
                            if f not in self.fstoi:
                                new_n = len(self.fstoi)
                                self.fstoi[f] = new_n
                                self.fitos[new_n] = f
                            f_n = self.fstoi[f]

                            # Next map it to a one hot index
                            if f_n not in self.multi2idx:
                                new_n = len(self.multi2idx)
                                self.multi2idx[f_n] = new_n
                                self.idx2multi[new_n] = f

                            fs_n.append(f_n)
                            self.cnames2fis[fn].add(f_n)
                        multifeats_p.append(fs_n)
                    else:
                        if fn == "lemma":
                            # Lowercase lemmas
                            f = f.lower()
                        if not f:
                            f = UNK_TOKEN
                        else:
                            f = f"{fn}:{f}"
                        if f not in self.fstoi:
                            new_n = len(self.fstoi)
                            self.fstoi[f] = new_n
                            self.fitos[new_n] = f
                        f_n = self.fstoi[f]
                        feats_p.append(f_n)
                        # Update feature name
                        self.cnames2fis[fn].add(f_n)
                doc_words.append(word_n)
                doc_feats.append(feats_p)
                doc_multifeats.append(multifeats_p)
            self.docs.append((doc_words, doc_feats, doc_multifeats))

    def is_multi(self, feat: str) -> bool:
        return self.ctypes[feat] == "multi"

    def __getitem__(self, i: int) -> tuple:
        ws = torch.as_tensor(self.words[i]).unsqueeze(1)
        fs = torch.as_tensor(self.fs[i])
        # These are multi-hot vectors with width equal to the number of
        # multi-hot features.
        mfs = torch.zeros((len(self.words[i]), len(self.mfis)), dtype=torch.bool)
        for w_i, word_mfeatures in enumerate(self.mfs[i]):
            for cat_f in word_mfeatures:
                for mfi in cat_f:
                    # Map to index
                    mfi_index = self.multi2idx[mfi]
                    mfs[w_i, mfi_index] = 1
        ls = torch.tensor(self.lengths[i])
        return ws, fs, mfs, ls, i

    def compute_cat_names(self, feat_str: str) -> tuple:
        cat_names = feat_str.strip().split("|")
        cat_names, ocs = zip(*[fn.split(";") for fn in cat_names])
        for oc in ocs:
            if oc not in {"open", "closed", "multi"}:
                raise RuntimeError(f"Invalid tag {oc}")
        ocn = [cn for cn, oc in zip(cat_names, ocs) if oc == "open"]
        ccn = [cn for cn, oc in zip(cat_names, ocs) if oc == "closed"]
        mcn = [cn for cn, oc in zip(cat_names, ocs) if oc == "multi"]
        cat_types = dict(zip(cat_names, ocs))
        cat_is = list(range(len(cat_names)))
        return cat_is, cat_names, cat_types, ocn, ccn, mcn

    def __len__(self) -> int:
        return len(self.words)

    # Added function
    def get_pair_text(self, pair_index: int) -> tuple:
        """Returns the premise, the hypothesis, and the label of a pair.

        Returns:
            tuple: premise, hypothesis, and label.
        """
        sentences = []
        for line in [self.text[2 * pair_index], self.text[2 * pair_index + 1]]:
            # Each token is "word|feature|feature|..."
            words = [token.split("|")[0] for token in line.strip().split(" ")]
            sentences.append(" ".join(words))
        premise, hypothesis = sentences
        return premise, hypothesis, self.target_itos[self.targets[pair_index]]

    # Adapted from the CoEx repository
    def to_sentence(self, toks: list, feats: dict, tok_feats_vocab: Optional[dict] = None) -> tuple:
        """
        Convert token-level feats to sentence feats. The features are named
        prefix:category:value, where the prefix is pre, hyp, or oth.
        """
        tokens = np.zeros(len(self.stoi), dtype=np.int64)
        encoder_uniques = []
        decoder_uniques = []

        encoder_tag_uniques = []
        decoder_tag_uniques = []

        tag_i = self.cstoi["tag"]

        other_features = []
        oth_names = [
            ("overlap25", "overlap"),
            ("overlap50", "overlap"),
            ("overlap75", "overlap"),
        ]

        for pair, featpair in zip(toks, feats["onehot"]):
            pair_counts = np.bincount(pair.ravel())
            tokens[: len(pair_counts)] += pair_counts

            enct = np.unique(pair[0])
            dect = np.unique(pair[1])

            encu = np.setdiff1d(enct, dect)
            decu = np.setdiff1d(dect, enct)
            both = np.intersect1d(enct, dect)
            encoder_uniques.append(enct)
            decoder_uniques.append(dect)

            # PoS
            enctag = np.unique(featpair[0, :, tag_i])
            dectag = np.unique(featpair[1, :, tag_i])

            enctag = enctag[enctag != -1]
            dectag = dectag[dectag != -1]

            encoder_tag_uniques.append(enctag)
            decoder_tag_uniques.append(dectag)

            # Compute degree of overlap in tokens (gt 50%)
            overlap = len(both) / (len(encu) + len(decu) + 1e-5)
            # TODO: Do overlap at various degrees
            other_features.append(
                (
                    overlap > 0.25,
                    overlap > 0.5,
                    overlap > 0.75,
                )
            )

        SKIP = {"a", "an", "the", "of", ".", ",", "UNK", "PAD"}
        if tok_feats_vocab is None:
            for s in SKIP:
                if s in self.stoi:
                    tokens[self.stoi[s]] = 0

            # Keep top tokens, use as features
            tokens_by_count = np.argsort(tokens)[::-1]
            tokens_by_count = tokens_by_count[: N_SENTENCE_FEATS]

            # Create feature dict
            # Token features
            tokens_stoi = {}
            for prefix in ["pre", "hyp"]:
                for t in tokens_by_count:
                    ts = self.itos[t]
                    t_prefixed = f"{prefix}:tok:{ts}"
                    tokens_stoi[t_prefixed] = len(tokens_stoi)
                # PoS
                for pos_i in self.cnames2fis["tag"]:
                    pos = self.fitos[pos_i].lower()
                    assert pos.startswith("tag:")
                    pos_prefixed = f"{prefix}:{pos}"
                    tokens_stoi[pos_prefixed] = len(tokens_stoi)

            # Other features
            for oth, oth_type in oth_names:
                oth_prefixed = f"oth:{oth_type}:{oth}"
                tokens_stoi[oth_prefixed] = len(tokens_stoi)

            tokens_itos = {v: k for k, v in tokens_stoi.items()}

            tok_feats_vocab = {
                "itos": tokens_itos,
                "stoi": tokens_stoi,
            }


        # Binary mask - encoder/decoder
        token_masks = np.zeros((len(toks), len(tok_feats_vocab["stoi"])), dtype=np.bool)
        for i, (encu, decu, enctagu, dectagu, oth) in enumerate(
            zip(
                encoder_uniques,
                decoder_uniques,
                encoder_tag_uniques,
                decoder_tag_uniques,
                other_features,
            )
        ):
            # Tokens
            for prefix, toks in [("pre", encu), ("hyp", decu)]:
                for t in toks:
                    ts = self.itos[t]
                    t_prefixed = f"{prefix}:tok:{ts}"
                    if t_prefixed in tok_feats_vocab["stoi"]:
                        ti = tok_feats_vocab["stoi"][t_prefixed]
                        token_masks[i, ti] = 1

            # PoS
            for prefix, tags in [("pre", enctagu), ("hyp", dectagu)]:
                for t in tags:
                    ts = self.fitos[t].lower()
                    t_prefixed = f"{prefix}:{ts}"
                    assert t_prefixed in tok_feats_vocab["stoi"]
                    ti = tok_feats_vocab["stoi"][t_prefixed]
                    token_masks[i, ti] = 1

            # Other features
            for (oth_name, oth_type), oth_u in zip(oth_names, oth):
                oth_prefixed = f"oth:{oth_type}:{oth_name}"
                oi = tok_feats_vocab["stoi"][oth_prefixed]
                token_masks[i, oi] = oth_u
        return token_masks, tok_feats_vocab
