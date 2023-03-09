"""Training for classic and INN Autoencoder.

Classes:
    Trainer: train and evaluate INN/classic Autoencoder.
"""


import torch
import torchvision
from tqdm.autonotebook import tqdm

from src.filemanager import save_model, save_numpy
from src.functionalities import get_device, get_model, get_optimizer, plot_image
from src.loss import LossTracker


class Trainer:
    """Class to train classic and INN autoencoders.

    Attributes:
        lat_dim (int): latent dimension.
        modelname (str): name of the model
        hyp_dict (dict): dictionary containing hyperparameters.
        tracker (class): loss tracker
        model (nn.Module): model to train and evaluate.
        optimizer (torch.optimizer): optimizer to use.
        scheduler (torch.lr_scheduler): scheduler to use.

    Methods:
        train_inn(torch.Dataloader, str) -> None: trains an INN Autoencoder.
        train_classic(torch.Dataloader, str) -> None: trains classic autoencoder.
        evaluate_inn(torch.Dataloader) -> float: evaluate INN autoencoder.
        evaluate_classic(torch.Dataloader) -> float: evaluate classic autoencoder.
    """

    def __init__(self, lat_dim: int, modelname: str, hyp_dict: dict) -> None:
        """Initialize the trainer.

        Args:
            lat_dim (int): latent dimension.
            modelname (str): name of the model.
            hyp_dict (dict): collections of hyperparameters.
        """
        self.lat_dim = lat_dim
        self.modelname = modelname
        self.hyp_dict = hyp_dict
        self.hyp_dict["device"] = get_device()
        self.tracker = LossTracker(self.lat_dim, self.hyp_dict, self.hyp_dict["device"])
        self.model = get_model(self.lat_dim, self.modelname, self.hyp_dict)
        self.optimizer = get_optimizer(self.model, self.hyp_dict)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=hyp_dict["milestones"], gamma=0.1
        )

    def train_inn(
        self, trainloader: torch.utils.data.DataLoader, subdir: str = ""
    ) -> None:
        """Trains an INN Autoencoder.

        Args:
            trainloader (torch.utils.data.DataLoader): dataloader for training.
            subdir (str): subdirectory to save the model. Defaults to None.
        """
        print(f"Start training for latent dimension: {self.lat_dim}")

        self.model.to(self.hyp_dict["device"])
        self.model.train()

        for epoch in range(self.hyp_dict["num_epoch"]):
            losses = torch.zeros(4, dtype=self.hyp_dict["dtype"])

            for data, target in tqdm(trainloader):
                data, target = data.to(self.hyp_dict["device"]), target.to(
                    self.hyp_dict["device"]
                )

                self.optimizer.zero_grad()

                lat_img = self.model(data)
                lat_shape = lat_img.shape
                lat_img = lat_img.view(lat_shape[0], -1)
                lat_img_zero = torch.cat(
                    [
                        lat_img[:, : self.lat_dim],
                        lat_img.new_zeros((lat_img[:, self.lat_dim :]).shape),
                    ],
                    dim=1,
                )
                lat_img_zero = lat_img_zero.view(lat_shape)
                rec = self.model(lat_img_zero, rev=True)

                batch_loss = self.tracker.inn_loss(data, rec, lat_img)
                batch_loss[0].backward()

                self.optimizer.step()
                self.scheduler.step()

                for i in range(4):
                    losses[i] += batch_loss[i].item()

            losses /= len(trainloader)
            self.tracker.update_inn_loss(losses, epoch, mode="train")

            print(
                f"Loss: {losses[0].cpu().detach():.3f} \t"
                + f"L_rec: {losses[1].cpu().detach():.3f} \t"
                + f"L_dist: {losses[2].cpu().detach():.3f} \t"
                + f"L_spar: {losses[3].cpu().detach():.3f}"
            )
            print("\n")
            print("-" * 80)
            print("\n")

        print("Finished training")

        self.model.to("cpu")
        save_model(self.model, f"{self.modelname}", subdir)

        train_loss_dict = self.tracker.get_loss(mode="train")
        save_numpy(
            train_loss_dict["total"].cpu().detach().numpy(),
            self.modelname + "_train_total",
            subdir,
        )
        save_numpy(
            train_loss_dict["rec"].cpu().detach().numpy(),
            self.modelname + "_train_rec",
            subdir,
        )
        save_numpy(
            train_loss_dict["dist"].cpu().detach().numpy(),
            self.modelname + "_train_dist",
            subdir,
        )
        save_numpy(
            train_loss_dict["sparse"].cpu().detach().numpy(),
            self.modelname + "_train_sparse",
            subdir,
        )

    def train_classic(
        self, trainloader: torch.utils.data.DataLoader, subdir: str = ""
    ) -> None:
        """Trains a classic Autoencoder.

        Args:
            trainloader (torch.utils.data.DataLoader): dataloader for training.
            subdir (str): subdirectory to save the model. Defaults to None.
        """
        # pylint: disable=W0107
        print(f"Start training for latent dimension: {self.lat_dim}")

        self.model.to(self.hyp_dict["device"])
        self.model.train()

        for epoch in range(self.hyp_dict["num_epoch"]):
            losses = torch.zeros(1, dtype=self.hyp_dict["dtype"])

            for data, target in tqdm(trainloader):
                data, target = data.to(self.hyp_dict["device"]), target.to(
                    self.hyp_dict["device"]
                )

                self.optimizer.zero_grad()
                rec = self.model(data)

                loss = self.tracker.l1_loss(data, rec)
                loss.backward()

                self.optimizer.step()
                self.scheduler.step()

                losses[0] += loss.item()

            losses /= len(trainloader)
            self.tracker.update_classic_loss(losses, epoch, mode="train")

            print(f"Loss: {losses[0].cpu().detach():.3f}")
            print("\n")
            print("-" * 80)
            print("\n")

        print("Finished training")

        self.model.to("cpu")
        save_model(self.model, f"{self.modelname}", subdir)

        train_loss_dict = self.tracker.get_loss(mode="train")
        save_numpy(
            train_loss_dict["rec"].cpu().detach().numpy(),
            self.modelname + "_train_rec",
            subdir,
        )

    def evaluate_inn(self, loader: torch.utils.data.DataLoader) -> float:
        """Evaluate reconstruction loss on INN Autoencoder.

        Args:
            loader (torch.utils.data.DataLoader): dataloader for evaluation.

        Returns:
            loss (float): mean reconstruction loss
        """
        self.model.to(self.hyp_dict["device"])
        self.model.eval()

        with torch.no_grad():
            loss = 0.0

            for data, target in tqdm(loader):
                data, target = data.to(self.hyp_dict["device"]), target.to(
                    self.hyp_dict["device"]
                )
                lat_img = self.model(data)
                lat_shape = lat_img.shape
                lat_img = lat_img.view(lat_shape[0], -1)
                lat_img_zero = torch.cat(
                    [
                        lat_img[:, : self.lat_dim],
                        lat_img.new_zeros((lat_img[:, self.lat_dim :]).shape),
                    ],
                    dim=1,
                )
                lat_img_zero = lat_img_zero.view(lat_shape)
                rec = self.model(lat_img_zero, rev=True)
                loss += self.tracker.l1_loss(data, rec).cpu().item()

            loss /= len(loader)

        return loss

    def evaluate_classic(self, loader: torch.utils.data.DataLoader) -> float:
        """Evaluate reconstruction loss on classic Autoencoder.

        Args:
            loader (torch.utils.data.DataLoader): dataloader for evaluation.

        Returns:
            loss (float): mean reconstruction loss
        """
        self.model.to(self.hyp_dict["device"])
        self.model.eval()

        with torch.no_grad():
            loss = 0.0

            for data, target in tqdm(loader):
                data, target = data.to(self.hyp_dict["device"]), target.to(
                    self.hyp_dict["device"]
                )
                rec = self.model(data)
                loss += self.tracker.l1_loss(data, rec).cpu().item()

            loss /= len(loader)

        return loss

    def plot_inn(
        self, loader: torch.utils.data.DataLoader, num_img: int, grid_row_size: int
    ) -> None:
        """Plot original, reconstructed and diffrence images for INN Autoencoder.

        Args:
            loader (torch.utils.data.DataLoader): dataloader for plotting
            num_img (int): The number of images to plot
            grid_row_size (int): number of images in a row of the grid
        """
        self.model.to(self.hyp_dict["device"])
        self.model.eval()

        data, target = next(iter(loader))
        data, target = data.to(self.hyp_dict["device"]), target.to(
            self.hyp_dict["device"]
        )

        lat_img = self.model(data)
        lat_shape = lat_img.shape
        lat_img = lat_img.view(lat_img.size(0), -1)
        lat_img_zero = torch.cat(
            [
                lat_img[:, : self.lat_dim],
                lat_img.new_zeros((lat_img[:, self.lat_dim :]).shape),
            ],
            dim=1,
        )
        lat_img_zero = lat_img_zero.view(lat_shape)
        rec = self.model(lat_img_zero, rev=True)
        diff_img = (data - rec + 1) / 2

        plot_image(
            torchvision.utils.make_grid(data[:num_img].cpu().detach(), grid_row_size),
            f"{self.modelname}_original",
        )
        plot_image(
            torchvision.utils.make_grid(rec[:num_img].cpu().detach(), grid_row_size),
            f"{self.modelname}_reconstructed",
        )
        plot_image(
            torchvision.utils.make_grid(
                diff_img[:num_img].cpu().detach(), grid_row_size
            ),
            f"{self.modelname}_difference",
        )
