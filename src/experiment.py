"""Run experiments in the paper.

Functions:
    experiment_wrapper(list) -> tuple: wrapper for running bottleneck experiments

Classes:
    Experiment: Run experiments in the paper
"""
import os
from typing import Union, cast

import torch

from src.dataloader import DATASET, get_loader
from src.filemanager import save_numpy
from src.functionalities import plot_curves
from src.settings import HYPERPARAMETER
from src.trainer import Trainer


class Experiment:
    """Class for running experiments.

    Attributes:
        modelname (list): name of the model to run experiments
        subdir (str): subdirectory for saving experiment results
        hyp_dict (dict): collection of hyperparameters
        bottleneck_loss (dict): train and test loss for each bottleneck size
        model_param_count (list):  number trainable parameters for each bottleneck size
        trainloader (torch.DataLoader): dataloader for trainset
        testloader (torch.DataLoader): dataloader for testset

    Methods:
        get_lat_dim_lst () -> list: return bottleneck sizes used in the experiments
        get_loss() -> dict: return train and test loss for all bottlenecks
        get_model_param_count() -> list: number of model trainable parameters per bottleneck
        run_experiment() -> None: run bottleneck experiment (entry point)
        run_inn_experiment() -> None: run bottleneck experiment for INN autoencoder
        run_classic_experiment() -> None: run bottleneck experiment for classic autoencoder
    """

    def __init__(self, modelname: str, subdir: str = "") -> None:
        """Initialize the experiment.

        Args:
            modelname (str): name of the model to run experiments
            subdir (str): subdirectory for saving experiment results. Defaults to "".
        """
        self.modelname = modelname
        self.subdir = subdir
        self.hyp_dict = HYPERPARAMETER[modelname]
        self.bottleneck_loss = {
            "train": torch.zeros(
                len(cast(list, self.hyp_dict["lat_dim_lst"])),
                dtype=cast(torch.dtype, self.hyp_dict["dtype"]),
            ),
            "test": torch.zeros(
                len(cast(list, self.hyp_dict["lat_dim_lst"])),
                dtype=cast(torch.dtype, self.hyp_dict["dtype"]),
            ),
        }
        self.model_param_count: list[int] = []

        trainset, testset = DATASET[self.modelname]

        if self.hyp_dict["subset"] is not None:
            trainset = torch.utils.data.Subset(
                trainset, cast(list[int], self.hyp_dict["subset"])
            )
            testset = torch.utils.data.Subset(
                testset, cast(list[int], self.hyp_dict["subset"])
            )

        self.trainloader = get_loader(
            trainset, cast(int, self.hyp_dict["batch_size"]), True
        )
        self.testloader = get_loader(
            testset, cast(int, self.hyp_dict["batch_size"]), False
        )

    def get_lat_dim_lst(self) -> list:
        """Return list of bottleneck dimensions used in the experiments.

        Returns:
            lat_dim_lst (list): list of bottleneck dimensions used in the experiments
        """
        lat_dim_lst = cast(list, self.hyp_dict["lat_dim_lst"])
        return lat_dim_lst

    def get_loss(self, mode: str) -> torch.Tensor:
        """Return train and test loss for all bottleneck sizes.

        Args:
            mode (str): return train or test loss. Options: "train", "test".

        Returns:
            bottleneck_loss (torch.Tensor): train and test loss for different bottleneck sizes
        """
        return self.bottleneck_loss[mode]

    def get_model_param_count(self) -> list[int]:
        """Return number of trainable parameters of autoencoder model.

        Returns:
            model_param_count (list): number of trainable parameters for each bottleneck size
        """
        return self.model_param_count

    def run_experiment(self) -> None:
        """Run classic or INN autoencoder bottleneck experiment."""
        if self.hyp_dict["INN"]:
            self.run_inn_experiment()
        else:
            self.run_classic_experiment()

        save_numpy(
            self.bottleneck_loss["train"].cpu().detach().numpy(),
            self.modelname + "_bottleneck_train_loss",
            self.subdir,
        )

        save_numpy(
            self.bottleneck_loss["test"].cpu().detach().numpy(),
            self.modelname + "_bottleneck_test_loss",
            self.subdir,
        )

    def run_inn_experiment(self) -> None:
        """Run INN autoencoder bottleneck experiment."""
        for idx, lat_dim in enumerate(cast(list[int], self.hyp_dict["lat_dim_lst"])):
            print("\n")
            print(f"Start Training with latent dimension: {lat_dim}")
            print("\n")

            save_path = (
                os.path.join(self.subdir, f"lat_dim_{lat_dim}")
                if self.subdir
                else f"lat_dim_{lat_dim}"
            )

            trainer = Trainer(lat_dim, self.modelname, self.hyp_dict)
            trainer.train_inn(self.trainloader, save_path)

            train_loss = trainer.evaluate_inn(self.trainloader)
            test_loss = trainer.evaluate_inn(self.testloader)
            self.bottleneck_loss["train"][idx] = train_loss
            self.bottleneck_loss["test"][idx] = test_loss
            self.model_param_count.append(trainer.count_model_param())

            trainer.plot_inn(self.testloader, 100, 10, save_path)

    def run_classic_experiment(
        self,
    ) -> None:
        """Run classic autoencoder bottleneck experiment."""
        for idx, lat_dim in enumerate(cast(list[int], self.hyp_dict["lat_dim_lst"])):
            print("\n")
            print(f"Start Training with latent dimension: {lat_dim}")
            print("\n")

            save_path = (
                os.path.join(self.subdir, f"lat_dim_{lat_dim}")
                if self.subdir
                else f"lat_dim_{lat_dim}"
            )

            trainer = Trainer(lat_dim, self.modelname, self.hyp_dict)
            trainer.train_classic(self.trainloader, save_path)

            train_loss = trainer.evaluate_classic(self.trainloader)
            test_loss = trainer.evaluate_classic(self.testloader)
            self.bottleneck_loss["train"][idx] = train_loss
            self.bottleneck_loss["test"][idx] = test_loss
            self.model_param_count.append(trainer.count_model_param())

            trainer.plot_classic(self.testloader, 100, 10, save_path)


def experiment_wrapper(
    model_lst: list[str], subdir: str = "", skip_exec: bool = False
) -> Union[tuple[list, list, list, list], int]:
    """Run bottleneck experiment and plotting results.

    Args:
        model_lst: list of model names to run bottleneck experiment
        subdir (str): subdirectory for saving experiment results. Defaults to "".
        skip_exec (bool): don't run bottleneck experiment (for unit tests)

    Returns:
        lat_dim_lst (list): bottleneck sizes used for each experiment
        train_loss_lst (list): trainset reconstruction loss
        test_loss_lst (list): testset reconstruction loss
    """
    if skip_exec:
        return 0

    lat_dim_lst = []
    train_loss_lst = []
    test_loss_lst = []
    num_param_lst = []

    for modelname in model_lst:
        exp = Experiment(modelname, subdir)
        exp.run_experiment()
        lat_dim_lst.append(exp.get_lat_dim_lst())
        train_loss_lst.append(exp.get_loss("train"))
        test_loss_lst.append(exp.get_loss("test"))
        num_param_lst.append(exp.get_model_param_count())

    bottleneck_plot_settings = {
        "names": model_lst,
        "x_label": "bottleneck size",
        "y_label": "reconstruction loss",
        "title": None,
    }

    num_param_plot_settings = {
        "names": model_lst,
        "x_label": "bottleneck size",
        "y_label": "number of parameters",
        "title": None,
    }

    plot_curves(
        lat_dim_lst,
        train_loss_lst,
        "train_bottleneck_loss",
        bottleneck_plot_settings,
        subdir,
    )
    plot_curves(
        lat_dim_lst,
        test_loss_lst,
        "test_bottleneck_loss",
        bottleneck_plot_settings,
        subdir,
    )
    plot_curves(
        lat_dim_lst,
        num_param_lst,
        "num_param_comparison",
        num_param_plot_settings,
        subdir,
    )

    return lat_dim_lst, train_loss_lst, test_loss_lst, num_param_lst
