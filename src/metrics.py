"""
Metrics for Khmer space injection (word segmentation)
"""
from typing import Tuple, Union
import torch


def space_precision_recall_f1(
    y_true: Union[torch.Tensor, list],
    y_pred: Union[torch.Tensor, list],
    ignore_index: int = -100
) -> Tuple[float, float, float]:
    """
    Compute precision/recall/F1 for "space" positions (label=1).

    Args:
        y_true: ground-truth labels (tensor or list)
        y_pred: predicted labels (tensor or list)
        ignore_index: ignored target value

    Returns:
        (precision, recall, f1)
    """
    # Convert to tensors
    y_true = torch.as_tensor(y_true)
    y_pred = torch.as_tensor(y_pred)

    # Flatten (supports [B, T] or [T])
    y_true = y_true.view(-1)
    y_pred = y_pred.view(-1)

    # Mask out ignored targets
    valid = (y_true != ignore_index)
    if valid.sum().item() == 0:
        return 0.0, 0.0, 0.0

    yt = y_true[valid]
    yp = y_pred[valid]

    # Positive class = 1 (space)
    tp = ((yp == 1) & (yt == 1)).sum().item()
    fp = ((yp == 1) & (yt == 0)).sum().item()
    fn = ((yp == 0) & (yt == 1)).sum().item()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return float(precision), float(recall), float(f1)


def exact_match_rate(seg_true: str, seg_pred: str) -> float:
    """
    Exact match rate at sentence level.

    Args:
        seg_true: gold segmented sentence
        seg_pred: predicted segmented sentence

    Returns:
        exact match (0.0 or 1.0)
    """
    def _normalize_spaces(s: str) -> str:
        # strip ends + collapse multiple spaces/tabs/newlines into single spaces
        return " ".join(s.strip().split())

    return 1.0 if _normalize_spaces(seg_true) == _normalize_spaces(seg_pred) else 0.0

