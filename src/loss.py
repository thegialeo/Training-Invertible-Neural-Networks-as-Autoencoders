"""Loss Functions used for training.

Functions:
    l1_loss(torch.Tensor, torch.Tensor, torch.dtype) -> torch.Tensor: L1 norm loss function
    l2_loss(torch.Tensor, torch.Tensor, torch.dtype) -> torch.Tensor: L2 norm loss function
    mmd_multiscale(torch.Tensor, torch.Tensor, torch.dtype, str) -> torch.Tensor: MMD loss function
"""

from typing import Optional

import torch

from src.functionalities import get_device


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


def mmd_multiscale(
    p_samples: torch.Tensor,
    q_samples: torch.Tensor,
    dtype: Optional[torch.dtype] = torch.float32,
    device: Optional[str] = get_device(),
) -> torch.Tensor:
    """Multi-scale kernel Maximum Mean Discrepancy loss function.

    Args:
        p_samples (torch.Tensor): samples from distribution P
        q_samples (torch.Tensor): samples from distribution Q
        dtype (torch.dtype): return type. Defaults to torch.float32.
        device (str): device. Defaults to get_device().

    Returns:
        loss (torch.Tensor): MMD loss between predictions and labels
    """
    p_samples, q_samples = p_samples.to(device), q_samples.to(device)

    step_pp = (
        (
            torch.mm(p_samples, p_samples.t())
            .diag()
            .unsqueeze(0)
            .expand_as(torch.mm(p_samples, p_samples.t()))
        ).t()
        + (
            torch.mm(p_samples, p_samples.t())
            .diag()
            .unsqueeze(0)
            .expand_as(torch.mm(p_samples, p_samples.t()))
        )
        - 2.0 * torch.mm(p_samples, p_samples.t())
    )
    step_qq = (
        (
            torch.mm(q_samples, q_samples.t())
            .diag()
            .unsqueeze(0)
            .expand_as(torch.mm(q_samples, q_samples.t()))
        ).t()
        + (
            torch.mm(q_samples, q_samples.t())
            .diag()
            .unsqueeze(0)
            .expand_as(torch.mm(q_samples, q_samples.t()))
        )
        - 2.0 * torch.mm(q_samples, q_samples.t())
    )
    step_pq = (
        (
            torch.mm(p_samples, p_samples.t())
            .diag()
            .unsqueeze(0)
            .expand_as(torch.mm(p_samples, p_samples.t()))
        ).t()
        + (
            torch.mm(q_samples, q_samples.t())
            .diag()
            .unsqueeze(0)
            .expand_as(torch.mm(q_samples, q_samples.t()))
        )
        - 2.0 * torch.mm(p_samples, q_samples.t())
    )

    sim_matrix_pp, sim_matrix_qq, sim_matrix_pq = (
        torch.zeros(step_pp.shape).to(device),
        torch.zeros(step_pp.shape).to(device),
        torch.zeros(step_pp.shape).to(device),
    )

    for bandwidth in [0.2, 0.5, 0.9, 1.3]:
        sim_matrix_pp += bandwidth**2 * (bandwidth**2 + step_pp) ** -1
        sim_matrix_qq += bandwidth**2 * (bandwidth**2 + step_qq) ** -1
        sim_matrix_pq += bandwidth**2 * (bandwidth**2 + step_pq) ** -1

    loss = torch.mean(sim_matrix_pp + sim_matrix_qq - 2.0 * sim_matrix_pq, dtype=dtype)

    return loss
