"""Loss functionalities used for training.

Classes:
    LossTracker: compute and tracks loss values.
"""

import torch

from src.functionalities import get_device


class LossTracker:
    """Class for computing and tracking loss values.

    Attributes:
        hyp_dict (dict): collection of hyperparameters
        train_loss_dict (dict): training loss values logger
        test_loss_dict (dict): test loss values logger

    Methods:
        l1_loss(Tensor, Tensor) -> Tensor: L1 norm loss function
        l2_loss(Tensor, Tensor) -> Tensor: L2 norm loss function
        mmd_multiscale(Tensor, Tensor) -> Tensor: MMD loss function
        inn_loss(Tensor, Tensor, Tensor) -> Tensor: INN Autoencoder loss function loss
        update_inn_loss(list[Tensor], int, str) -> None: track inn autoencoder loss
        update_classic_loss(Tensor, int, str) -> None: track classic autoencoder loss
    """

    def __init__(self, hyp_dict: dict) -> None:
        """Initialize loss tracker.

        Args:
            hyp_dict (dict): collection of hyperparameters
        """
        self.hyp_dict = hyp_dict
        if hyp_dict["INN"]:
            self.train_loss = {
                "total": torch.zeros(hyp_dict["num_epoch"]),
                "rec": torch.zeros(hyp_dict["num_epoch"]),
                "dist": torch.zeros(hyp_dict["num_epoch"]),
                "sparse": torch.zeros(hyp_dict["num_epoch"]),
            }
            self.test_loss = {
                "total": torch.zeros(hyp_dict["num_epoch"]),
                "rec": torch.zeros(hyp_dict["num_epoch"]),
                "dist": torch.zeros(hyp_dict["num_epoch"]),
                "sparse": torch.zeros(hyp_dict["num_epoch"]),
            }
        else:
            self.train_loss = {"total": torch.zeros(hyp_dict["num_epoch"])}
            self.test_loss = {"total": torch.zeros(hyp_dict["num_epoch"])}

    def l1_loss(self, pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """L1-norm loss function (least absolute deviations).

        Args:
            pred (torch.Tensor): batch of network predictions
            label (torch.Tensor): batch of labels

        Returns:
            loss (torch.Tensor): L1-norm loss between predictions and labels
        """
        loss = torch.mean(torch.abs(pred - label), dtype=self.hyp_dict["dtype"])
        return loss

    def l2_loss(self, pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """L2-norm loss function (least squares error).

        Args:
            pred (torch.Tensor): batch of network predictions
            label (torch.Tensor): batch of labels

        Returns:
            loss (torch.Tensor): L2-norm loss between predictions and labels
        """
        loss = torch.mean(torch.pow(pred - label, 2), dtype=self.hyp_dict["dtype"])
        return loss

    def mmd_multiscale(
        self, p_samples: torch.Tensor, q_samples: torch.Tensor
    ) -> torch.Tensor:
        """Multi-scale kernel Maximum Mean Discrepancy loss function.

        Args:
            p_samples (torch.Tensor): samples from distribution P
            q_samples (torch.Tensor): samples from distribution Q

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

        loss = torch.mean(
            sim_matrix_pp + sim_matrix_qq - 2.0 * sim_matrix_pq,
            dtype=self.hyp_dict["dtype"],
        )

        return loss

    def inn_loss(
        self, inp: torch.Tensor, rec: torch.Tensor, lat: torch.Tensor
    ) -> list[torch.Tensor]:
        """Compute INN Autoencoder loss.

        Args:
            inp (torch.Tensor): input image
            rec (torch.Tensor): reconstructed image
            lat (torch.Tensor): latent vector

        Returns:
            list[torch.Tensor]: total, reconstruction, distribution and sparse loss
        """
        base_dist = torch.normal(0, 1, size=(lat.size(0), self.hyp_dict["lat_dim"]))

        l_rec = self.hyp_dict["a_rec"] * self.l1_loss(inp, rec)
        l_dist = self.hyp_dict["a_dist"] * self.mmd_multiscale(
            lat[:, : self.hyp_dict["lat_dim"]], base_dist
        )
        l_sparse = self.hyp_dict["a_sparse"] * self.l2_loss(
            lat[:, self.hyp_dict["lat_dim"] :],
            torch.zeros(lat.size(0), lat.size(1) - self.hyp_dict["lat_dim"]),
        )

        l_total = l_rec + l_dist + l_sparse

        return [l_total, l_rec, l_dist, l_sparse]

    def update_inn_loss(self, loss: list[torch.Tensor], epoch: int, mode: str) -> None:
        """Track INN Autoencoder loss.

        Args:
            loss (list[torch.Tensor]): loss values to be tracked.
            epoch (int): current epoch
            mode (str): training or testing mode. Options: "train", "test".

        Raises:
            ValueError: if mode is not 'train' or 'test'
        """
        if mode == "train":
            self.train_loss["total"][epoch] = loss[0]
            self.train_loss["rec"][epoch] = loss[1]
            self.train_loss["dist"][epoch] = loss[2]
            self.train_loss["sparse"][epoch] = loss[3]
        elif mode == "test":
            self.test_loss["total"][epoch] = loss[0]
            self.test_loss["rec"][epoch] = loss[1]
            self.test_loss["dist"][epoch] = loss[2]
            self.test_loss["sparse"][epoch] = loss[3]
        else:
            raise ValueError(
                f"Mode {mode} is not supported. Options are 'train' and 'test'."
            )

    def update_classic_loss(self, loss: torch.Tensor, epoch: int, mode: str) -> None:
        """Track classic autoencoder loss.

        Args:
            loss (torch.Tensor): loss values to be tracked.
            epoch (int): current epoch
            mode (str): training or testing mode. Options: "train", "test".

        Raises:
            ValueError: if mode is not 'train' or 'test'
        """
        if mode == "train":
            self.train_loss["total"][epoch] = loss
        elif mode == "test":
            self.test_loss["total"][epoch] = loss
        else:
            raise ValueError(
                f"Mode {mode} is not supported. Options are 'train' and 'test'."
            )
