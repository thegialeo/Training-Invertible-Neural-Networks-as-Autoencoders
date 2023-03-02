"""Loss Functions used for training.

Functions:
    l1_loss(Tensor, Tensor, dtype) -> Tensor: L1 norm loss function
    l2_loss(Tensor, Tensor, dtype) -> Tensor: L2 norm loss function
    mmd_multiscale(Tensor, Tensor, dtype, str) -> Tensor: MMD loss function
    inn_loss(Tensor, Tensor, Tensor, dict, dtype) -> Tensor: INN Autoencoder loss function loss
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
) -> torch.Tensor:
    """Multi-scale kernel Maximum Mean Discrepancy loss function.

    Args:
        p_samples (torch.Tensor): samples from distribution P
        q_samples (torch.Tensor): samples from distribution Q
        dtype (torch.dtype): return type. Defaults to torch.float32.

    Returns:
        loss (torch.Tensor): MMD loss between predictions and labels
    """
    device = get_device()
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


def inn_loss(
    inp: torch.Tensor,
    rec: torch.Tensor,
    lat: torch.Tensor,
    hyp_dict: dict,
    dtype: Optional[torch.dtype] = torch.float32,
) -> list[torch.Tensor]:
    """Compute INN Autoencoder loss.

    Args:
        inp (torch.Tensor): input image
        rec (torch.Tensor): reconstructed image
        lat (torch.Tensor): latent vector
        hyp_dict (dict): collection of hyperparameters (latent dimension and loss term coefficients)
        dtype (torch.dtype): return type. Defaults to torch.float32.

    Returns:
        list[torch.Tensor]: total, reconstruction, distribution and sparse loss
    """
    base_dist = torch.normal(0, 1, size=(lat.size(0), hyp_dict["lat_dim"]))

    l_rec = hyp_dict["a_rec"] * l1_loss(inp, rec, dtype=dtype)
    l_dist = hyp_dict["a_dist"] * mmd_multiscale(
        lat[:, : hyp_dict["lat_dim"]], base_dist, dtype=dtype
    )
    l_sparse = hyp_dict["a_sparse"] * l2_loss(
        lat[:, hyp_dict["lat_dim"] :],
        torch.zeros(lat.size(0), lat.size(1) - hyp_dict["lat_dim"]),
        dtype=dtype,
    )

    l_total = l_rec + l_dist + l_sparse

    return [l_total, l_rec, l_dist, l_sparse]
