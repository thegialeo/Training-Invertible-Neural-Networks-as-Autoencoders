"""Loss Functions used for training.

Functions:
    l1_loss(torch.Tensor, torch.Tensor, torch.dtype) -> torch.Tensor
"""

from typing import Optional

import torch


def l1_loss(
    pred: torch.Tensor, label: torch.Tensor, dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    """L1-norm loss function (least absolute deviations).

    Args:
        pred (torch.Tensor): batch of network predictions
        label (torch.Tensor): batch of labels
        dtype (torch.dtype): return type

    Returns:
        loss (torch.Tensor): L1-norm loss between predictions and labels
    """
    loss = torch.mean(torch.abs(pred - label), dtype=dtype)
    return loss
