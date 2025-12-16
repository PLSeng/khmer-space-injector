"""
Metrics for Khmer space injection (word segmentation)
"""

# TODO: PEN Virak
def space_precision_recall_f1(y_true, y_pred, ignore_index: int = -100):
    """
    Compute precision/recall/F1 for "space" positions (label=1).

    Args:
        y_true: ground-truth labels (tensor or list)
        y_pred: predicted labels (tensor or list)
        ignore_index: ignored target value

    Returns:
        (precision, recall, f1)
    """
    pass


# TODO: PEN Virak
def exact_match_rate(seg_true: str, seg_pred: str) -> float:
    """
    Exact match rate at sentence level.

    Args:
        seg_true: gold segmented sentence
        seg_pred: predicted segmented sentence

    Returns:
        exact match (0.0 or 1.0) or averaged outside
    """
    pass

