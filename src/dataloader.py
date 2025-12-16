"""
Data loading and preprocessing for Khmer space injection
"""
import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict


class KhmerDataset(Dataset):
    """
    Dataset for Khmer text with space injection labels
    """
    
    
    def __init__(self, texts, labels=None, char_to_index=None, max_length=128):
        """
        Initialize dataset
        
        Args:
            texts: List of input texts
            labels: List of label sequences (1 for space, 0 for no space)
            char_to_index: Dictionary mapping characters to indices
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.max_length = max_length
        # Build vocabulary if not provided
        if char_to_index is None:
            char_set = set()
            for text in texts:
                char_set.update(text)
            char_to_index = {char: idx + 2 for idx, char in enumerate(sorted(char_set))}
            char_to_index['<PAD>'] = 0
            char_to_index['<UNK>'] = 1
        self.char_to_index = char_to_index
        self.index_to_char = {idx: char for char, idx in char_to_index.items()}
        
    def __len__(self):
        """
        Get dataset length
        
        Returns:
            Length of the dataset
        """
        return len(self.texts)
    
    def __getitem__(self, idx: int):
        """
        Get item at index
        
        Args:
            idx: Index
            
        Returns:
            Tuple of (input_tensor, label_tensor)
        """
        text = self.texts[idx]
        
        # Convert characters to indices
        input_ids = [self.char_to_index.get(c, self.char_to_index['<UNK>']) for c in text]
        
        if self.labels is not None:
            label_seq = self.labels[idx]  # should be list of 0/1 with len = len(text) - 1
            if len(label_seq) != len(text) - 1:
                raise ValueError(f"Label length mismatch for index {idx}: text len {len(text)}, labels {len(label_seq)}")
        else:
            # For inference mode, no labels
            label_seq = None
        
        # Truncate if too long
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            if label_seq is not None:
                label_seq = label_seq[:self.max_length - 1]
        
        # Convert to tensors
        input_tensor = torch.tensor(input_ids, dtype=torch.long)
        
        if label_seq is not None:
            label_tensor = torch.tensor(label_seq, dtype=torch.long)
            return input_tensor, label_tensor
        else:
            return input_tensor, torch.tensor([])  # empty label for inference

def collate_fn(batch):
    """
    Collate function to pad sequences in a batch
    """
    inputs, labels = zip(*batch)
    
    # Pad inputs
    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=0)
    
    # Pad labels (if any exist)
    if labels[0].numel() > 0:
        # append an ignore token so label length == input length (labels indicate space AFTER each char)
        new_labels = [torch.cat([l, torch.tensor([-100], dtype=torch.long)]) for l in labels]
        labels_padded = pad_sequence(new_labels, batch_first=True, padding_value=-100)  # -100 is ignore_index for loss
    else:
        labels_padded = None
    
    return inputs_padded, labels_padded
def create_dataloader(texts, labels=None, char_to_index=None, batch_size=32, max_length=128, shuffle=True, num_workers=0):
    """
    Create DataLoader for training or inference
    
    Args:
        texts: List of input texts
        labels: List of label sequences
        char_to_index: Dictionary mapping characters to indices
        batch_size: Batch size
        max_length: Maximum sequence length
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes
        
    Returns:
        DataLoader instance
    """
    dataset = KhmerDataset(texts, labels, char_to_index, max_length)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    return dataloader, dataset.char_to_index  # return vocab for potential reuse


def load_data(file_path: str,
              batch_size=None,
              max_length=128,
              char_to_index=None,
              shuffle=True,
              num_workers=0,
              skip_invalid=True,
              return_vocab=False):
    """
    Load data from file
    
    Expected file format: Each line should contain text and space-separated labels
    Example: "text_string\t0 1 0 1 0"
    
    Args:
        file_path: Path to data file
        
    Returns:
        Tuple of (texts, labels)
    """
    texts = []
    labels = []

    if os.path.isdir(file_path):
        txt_files = [os.path.join(file_path, f) for f in os.listdir(file_path) if f.endswith('.txt')]
    else:
        matches = glob.glob(file_path)
        if matches:
            txt_files = matches
        elif os.path.isfile(file_path):
            txt_files = [file_path]
        else:
            raise ValueError(f"Invalid path or pattern: {file_path}")

    skipped = 0
    total_lines = 0

    for fp in txt_files:
        with open(fp, 'r', encoding='utf-8') as f:
            for line in f:
                total_lines += 1
                line = line.strip()
                if not line:
                    continue

                words = line.split()
                if not words:
                    continue

                continuous_text = ''.join(words)

                # === CORRECT LABEL CONSTRUCTION ===
                label_seq = []
                for word in words[:-1]:
                    label_seq.extend([0] * (len(word) - 1))
                    label_seq.append(1)  # space after this word
                if words:
                    label_seq.extend([0] * (len(words[-1]) - 1))

                expected_len = len(continuous_text) - 1
                if len(label_seq) != expected_len:
                    if skip_invalid:
                        skipped += 1
                        continue
                    else:
                        raise ValueError(f"Label length mismatch: {len(label_seq)} vs expected {expected_len}")

                texts.append(continuous_text)
                labels.append(label_seq)

    if not texts:
        raise ValueError("No valid data found")

    if skipped:
        print(f"Skipped {skipped} invalid lines out of {total_lines}")

    if batch_size is not None:
        dataloader, vocab = create_dataloader(
            texts, labels, char_to_index,
            batch_size=batch_size, max_length=max_length,
            shuffle=shuffle, num_workers=num_workers
        )
        if return_vocab:
            return dataloader, vocab
        return dataloader

    return texts, labels, char_to_index
