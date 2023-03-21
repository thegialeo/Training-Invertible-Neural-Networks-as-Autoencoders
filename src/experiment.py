"""Run experiments in the paper.

Functions:
    experiment_wrapper(list) -> tuple: wrapper for running bottleneck experiments

Classes:
    Experiment: Run experiments in the paper
"""
import os
from typing import cast

import torch

from src.dataloader import DATASET, get_loader
from src.settings import HYPERPARAMETER
from src.trainer import Trainer


class Experiment:
    """Class for running experiments.

    Attributes:
        modelname: name of the model to run experiments
        hyp_dict (dict): collection of hyperparameters
        bottleneck_loss (dict): train and test loss for different bottleneck sizes
        subdir (str): subdirectory for saving experiment results
        trainloader: dataloader for trainset
        testloader: dataloader for testset

    Methods:
        run_inn_experiment() -> None: run bottleneck experiment for INN autoencoder
        run_classic_experiment() -> None: run bottleneck experiment for classic autoencoder
        get_loss() -> dict: return train and test loss for all bottlenecks
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

        trainset, testset = DATASET[self.modelname]
        self.trainloader = get_loader(
            trainset, cast(int, self.hyp_dict["batch_size"]), True
        )
        self.testloader = get_loader(
            testset, cast(int, self.hyp_dict["batch_size"]), False
        )

    def get_loss(self, key: str) -> torch.Tensor:
        """Return train and test loss for all bottleneck sizes.

        Args:
            key (str): dictionary key. Options: 'train' and 'test'

        Returns:
            bottleneck_loss (torch.Tensor): train and test loss for different bottleneck sizes
        """
        return self.bottleneck_loss[key]

    def get_lat_dim_lst(self) -> list:
        """Return list of bottleneck dimensions used in the experiments.

        Returns:
            lat_dim_lst (list): list of bottleneck dimensions used in the experiments
        """
        lat_dim_lst = cast(list, self.hyp_dict["lat_dim_lst"])
        return lat_dim_lst

    def run_experiment(self) -> None:
        """Run classic or INN autoencoder bottleneck experiment."""
        if self.hyp_dict["INN"]:
            self.run_inn_experiment()
        else:
            self.run_classic_experiment()

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

            trainer.plot_classic(self.testloader, 100, 10, save_path)


def experiment_wrapper(model_lst: list[str]) -> tuple[list, list, list]:
    """Run experiment wrapper.

    Args:
        model_lst: list of model names to run bottleneck experiment

    Returns:
        lat_dim_lst (list): bottleneck sizes used for each experiment
        train_loss_lst (list): trainset reconstruction loss
        test_loss_lst (list): testset reconstruction loss
    """
    lat_dim_lst = []
    train_loss_lst = []
    test_loss_lst = []

    for modelname in model_lst:
        exp = Experiment(modelname)
        exp.run_experiment()
        lat_dim_lst.append(exp.get_lat_dim_lst())
        train_loss_lst.append(exp.get_loss("train"))
        test_loss_lst.append(exp.get_loss("test"))

    return lat_dim_lst, train_loss_lst, test_loss_lst
