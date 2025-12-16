"""
Data loading and preprocessing for Khmer space injection
"""

import os
import glob
from typing import List, Tuple, Optional, Dict

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from src.utils import build_vocab, char_to_idx


# ======================================================
# CONFIGURATION (change here if dataset location changes)
# ======================================================
DEFAULT_DATA_DIR = "data"   # e.g. "data", "/mnt/datasets/khmer", "../data"


# ======================================================
# Dataset
# ======================================================
class KhmerDataset(Dataset):
    """
    Dataset for Khmer text with space injection labels
    """

    def __init__(
        self,
        texts: List[str],
        labels: Optional[List[List[int]]] = None,
        char_to_index: Optional[Dict[str, int]] = None,
        max_length: int = 128,
    ):
        self.texts = texts
        self.labels = labels
        self.max_length = max_length

        if char_to_index is None:
            char_to_index, _ = build_vocab(texts)

        self.char_to_index = char_to_index

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        text = self.texts[idx]

        input_ids = [char_to_idx(c, self.char_to_index) for c in text]

        label_seq = None
        if self.labels is not None:
            label_seq = self.labels[idx]
            if len(label_seq) != len(text) - 1:
                raise ValueError(
                    f"Label length mismatch for index {idx}: text len {len(text)}, labels {len(label_seq)}"
                )

        if len(input_ids) > self.max_length:
            input_ids = input_ids[: self.max_length]
            if label_seq is not None:
                label_seq = label_seq[: self.max_length - 1]

        input_tensor = torch.tensor(input_ids, dtype=torch.long)

        if label_seq is not None:
            label_tensor = torch.tensor(label_seq, dtype=torch.long)
            return input_tensor, label_tensor

        return input_tensor, torch.tensor([])


# ======================================================
# Collate
# ======================================================
def collate_fn(batch):
    inputs, labels = zip(*batch)

    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=0)

    if labels[0].numel() > 0:
        new_labels = [torch.cat([l, torch.tensor([-100])]) for l in labels]
        labels_padded = pad_sequence(new_labels, batch_first=True, padding_value=-100)
    else:
        labels_padded = None

    return inputs_padded, labels_padded


def create_dataloader(
    texts: List[str],
    labels: Optional[List[List[int]]] = None,
    char_to_index: Optional[Dict[str, int]] = None,
    batch_size: int = 32,
    max_length: int = 128,
    shuffle: bool = True,
    num_workers: int = 0,
):
    dataset = KhmerDataset(texts, labels, char_to_index, max_length)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    return dataloader, dataset.char_to_index


# ======================================================
# Pair loading helpers
# ======================================================
def _strip_all_whitespace(s: str) -> str:
    return "".join(s.split())


def _build_labels_from_segmented_words(words: List[str]) -> Tuple[str, List[int]]:
    continuous_text = "".join(words)

    label_seq: List[int] = []
    for w in words[:-1]:
        label_seq.extend([0] * (len(w) - 1))
        label_seq.append(1)
    if words:
        label_seq.extend([0] * (len(words[-1]) - 1))

    return continuous_text, label_seq


# ======================================================
# Public loader
# ======================================================
def load_data(
    data_dir: str = DEFAULT_DATA_DIR,
    batch_size: Optional[int] = None,
    max_length: int = 128,
    char_to_index: Optional[Dict[str, int]] = None,
    shuffle: bool = True,
    num_workers: int = 0,
    skip_invalid: bool = True,
    return_vocab: bool = False,
):
    """
    Load paired Khmer dataset from folder:
        *_orig.txt  (no spaces)
        *_seg.txt   (with spaces)
    """

    if not os.path.isdir(data_dir):
        raise ValueError(f"data_dir not found: {data_dir}")

    seg_files = sorted(glob.glob(os.path.join(data_dir, "*_seg.txt")))
    if not seg_files:
        raise ValueError(f"No *_seg.txt files found in: {data_dir}")

    texts: List[str] = []
    labels: List[List[int]] = []

    skipped = 0
    total = 0

    for seg_path in seg_files:
        orig_path = seg_path.replace("_seg.txt", "_orig.txt")
        if not os.path.isfile(orig_path):
            if skip_invalid:
                skipped += 1
                continue
            raise FileNotFoundError(f"Missing paired file: {orig_path}")

        with open(seg_path, "r", encoding="utf-8") as f_seg, open(orig_path, "r", encoding="utf-8") as f_orig:
            for seg_line, orig_line in zip(f_seg, f_orig):
                total += 1
                seg_line = seg_line.strip()
                orig_line = orig_line.strip()

                if not seg_line or not orig_line:
                    continue

                words = seg_line.split()
                cont_text, label_seq = _build_labels_from_segmented_words(words)

                if _strip_all_whitespace(seg_line) != _strip_all_whitespace(orig_line):
                    if skip_invalid:
                        skipped += 1
                        continue
                    raise ValueError("orig/seg mismatch after whitespace removal")

                if len(label_seq) != len(cont_text) - 1:
                    if skip_invalid:
                        skipped += 1
                        continue
                    raise ValueError("Label length mismatch")

                texts.append(cont_text)
                labels.append(label_seq)

    if not texts:
        raise ValueError("No valid data found")

    if skipped:
        print(f"[INFO] Skipped {skipped} invalid lines")

    if char_to_index is None:
        char_to_index, _ = build_vocab(texts)

    if batch_size is not None:
        dataloader, vocab = create_dataloader(
            texts, labels, char_to_index,
            batch_size, max_length, shuffle, num_workers
        )
        return (dataloader, vocab) if return_vocab else dataloader

    return texts, labels, char_to_index
