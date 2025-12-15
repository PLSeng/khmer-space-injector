"""
Data loading and preprocessing for Khmer space injection
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional
import numpy as np


class KhmerDataset(Dataset):
    """
    Dataset for Khmer text with space injection labels
    """
    
    def __init__(
        self,
        texts: List[str],
        labels: Optional[List[List[int]]] = None,
        char_to_index: Optional[dict] = None,
        max_length: int = 128
    ):
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
        self.char_to_index = char_to_index or {}
        self.max_length = max_length
        
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Get item at index
        
        Args:
            idx: Index
            
        Returns:
            Tuple of (input_tensor, label_tensor)
        """
        text = self.texts[idx]
        
        # Convert text to indices
        indices = [
            self.char_to_index.get(char, self.char_to_index.get('<UNK>', 0))
            for char in text[:self.max_length]
        ]
        
        # Pad sequence
        padding_length = max(0, self.max_length - len(indices))
        indices = indices + [self.char_to_index.get('<PAD>', 0)] * padding_length
        
        input_tensor = torch.LongTensor(indices)
        
        if self.labels is not None:
            label = self.labels[idx][:self.max_length]
            # Pad labels
            label = label + [0] * padding_length
            label_tensor = torch.LongTensor(label)
            return input_tensor, label_tensor
        
        return input_tensor, None


def create_dataloader(
    texts: List[str],
    labels: Optional[List[List[int]]] = None,
    char_to_index: Optional[dict] = None,
    batch_size: int = 32,
    max_length: int = 128,
    shuffle: bool = True,
    num_workers: int = 0
) -> DataLoader:
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
    dataset = KhmerDataset(
        texts=texts,
        labels=labels,
        char_to_index=char_to_index,
        max_length=max_length
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
    
    return dataloader


def load_data(file_path: str) -> Tuple[List[str], List[List[int]]]:
    """
    Load data from file
    
    Args:
        file_path: Path to data file
        
    Returns:
        Tuple of (texts, labels)
    """
    texts = []
    labels = []
    
    # TODO: Implement data loading logic based on your file format
    # This is a placeholder implementation
    
    return texts, labels
