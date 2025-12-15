"""
Utility functions for Khmer space injection RNN model
"""

import numpy as np
import torch
from typing import List, Tuple, Optional


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility
    
    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def char_to_idx(char: str, char_to_index: dict) -> int:
    """
    Convert character to index
    
    Args:
        char: Character to convert
        char_to_index: Dictionary mapping characters to indices
        
    Returns:
        Index of the character
    """
    return char_to_index.get(char, char_to_index.get('<UNK>', 0))


def idx_to_char(idx: int, index_to_char: dict) -> str:
    """
    Convert index to character
    
    Args:
        idx: Index to convert
        index_to_char: Dictionary mapping indices to characters
        
    Returns:
        Character at the index
    """
    return index_to_char.get(idx, '<UNK>')


def build_vocab(texts: List[str]) -> Tuple[dict, dict]:
    """
    Build vocabulary from texts
    
    Args:
        texts: List of text strings
        
    Returns:
        Tuple of (char_to_index, index_to_char) dictionaries
    """
    chars = set()
    for text in texts:
        chars.update(text)
    
    # Sort for consistency
    chars = sorted(list(chars))
    
    # Add special tokens
    chars = ['<PAD>', '<UNK>'] + chars
    
    char_to_index = {char: idx for idx, char in enumerate(chars)}
    index_to_char = {idx: char for idx, char in enumerate(chars)}
    
    return char_to_index, index_to_char


def save_model(model, path: str) -> None:
    """
    Save model to disk
    
    Args:
        model: PyTorch model to save
        path: Path to save the model
    """
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def load_model(model, path: str) -> None:
    """
    Load model from disk
    
    Args:
        model: PyTorch model instance
        path: Path to load the model from
    """
    model.load_state_dict(torch.load(path))
    print(f"Model loaded from {path}")
