"""
Utility functions for Khmer space injection RNN model
"""

# TODO: HengHeng
def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility
    
    Args:
        seed: Random seed value
    """
    pass


# TODO: Lyhourt
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


# TODO: Lyhourt
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

# TODO: HengHeng
def build_vocab(texts):
    """
    Build vocabulary from texts
    
    Args:
        texts: List of text strings
        
    Returns:
        Tuple of (char_to_index, index_to_char) dictionaries
    """
    pass


# TODO: SOL Visal
def save_model(model, path: str) -> None:
    """
    Save model to disk
    
    Args:
        model: PyTorch model to save
        path: Path to save the model
    """
    pass


# TODO: SOL Visal
def load_model(model, path: str, device: str = 'cpu') -> None:
    """
    Load model from disk
    
    Args:
        model: PyTorch model instance
        path: Path to load the model from
        device: Device to load the model on ('cpu' or 'cuda')
    """
    pass
