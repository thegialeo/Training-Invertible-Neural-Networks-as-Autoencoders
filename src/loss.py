"""Loss Functions used for training.

Functions:
    l1_loss(torch.Tensor, torch.Tensor, torch.dtype) -> torch.Tensor: L1 norm loss function
    l2_loss(torch.Tensor, torch.Tensor, torch.dtype) -> torch.Tensor: L2 norm loss function
"""

from typing import Optional

import torch


def l1_loss(
    pred: torch.Tensor,
    label: torch.Tensor,
    dtype: Optional[torch.dtype] = torch.float32,
) -> torch.Tensor:
    """L1-norm loss function (least absolute deviations).

    Args:
        pred (torch.Tensor): batch of network predictions
        label (torch.Tensor): batch of labels
        dtype (torch.dtype): return type. Defaults to torch.float32.

    Returns:
        loss (torch.Tensor): L1-norm loss between predictions and labels
    """
    loss = torch.mean(torch.abs(pred - label), dtype=dtype)
    return loss


def l2_loss(
    pred: torch.Tensor,
    label: torch.Tensor,
    dtype: Optional[torch.dtype] = torch.float32,
) -> torch.Tensor:
    """L2-norm loss function (least squares error).

    Args:
        pred (torch.Tensor): batch of network predictions
        label (torch.Tensor): batch of labels
        dtype (torch.dtype): return type. Defaults to torch.float32.

    Returns:
        loss (torch.Tensor): L2-norm loss between predictions and labels
    """
    loss = torch.mean(torch.pow(pred - label, 2), dtype=dtype)
    return loss
