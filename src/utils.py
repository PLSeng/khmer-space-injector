"""
Utility functions for Khmer space injection RNN model
"""
import argparse
import random
import numpy as np
import torch

def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior (may reduce performance slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def char_to_idx(char: str, char_to_index: dict) -> int:
    """
    Convert character to index
    
    Args:
        char: Character to convert
        char_to_index: Dictionary mapping characters to indices
        
    Returns:
        Index of the character
    """
    if char not in char_to_index:
        raise ValueError(f"Character '{char}' not found in char_to_index dictionary")
    return char_to_index[char]

def idx_to_char(idx: int, index_to_char: dict) -> str:
    """
    Convert index to character
    
    Args:
        idx: Index to convert
        index_to_char: Dictionary mapping indices to characters
        
    Returns:
        Character at the index
    """
    if idx not in index_to_char:
        raise ValueError(f"Index '{idx}' not found in index_to_char dictionary")
    return index_to_char[idx]

def build_vocab(texts):
    """
    Build vocabulary from texts
    
    Args:
        texts: List of text strings
        
    Returns:
        Tuple of (char_to_index, index_to_char) dictionaries
    """
    # Collect unique characters
    chars = set()
    for text in texts:
        chars.update(text)

    # Sort characters for consistent indexing
    chars = sorted(chars)

    char_to_index = {char: idx for idx, char in enumerate(chars)}
    index_to_char = {idx: char for char, idx in char_to_index.items()}

    return char_to_index, index_to_char


def save_model(model, path: str) -> None:
    """
    Save model to disk
    
    Args:
        model: PyTorch model to save
        path: Path to save the model
    """
    torch.save(model.state_dict(), path)


def load_model(model, path: str, device: str = 'cpu') -> None:
    """
    Load model from disk
    
    Args:
        model: PyTorch model instance
        path: Path to load the model from
        device: Device to load the model on ('cpu' or 'cuda')
    """
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()


def str2bool(v):
    # accepts True/False from wandb like "--flag=True"
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    s = str(v).strip().lower()
    if s in ("1", "true", "t", "yes", "y", "on"):
        return True
    if s in ("0", "false", "f", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean expected, got: {v}")
