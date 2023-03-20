"""Run experiments in the paper.

Classes:
    Experiment: Run experiments in the paper
"""
import os
from typing import Union, cast

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
        self.hyp_dict: dict[Union[list[int], str]] = HYPERPARAMETER[modelname]
        self.bottleneck_loss = {
            "train": torch.zeros(
                len(self.hyp_dict["lat_dim_lst"]),
                dtype=self.hyp_dict["dtype"],
            ),
            "test": torch.zeros(
                len(self.hyp_dict["lat_dim_lst"]),
                dtype=self.hyp_dict["dtype"],
            ),
        }

        trainset, testset = DATASET[self.modelname]
        self.trainloader = get_loader(
            trainset, cast(int, self.hyp_dict["batch_size"]), True
        )
        self.testloader = get_loader(
            testset, cast(int, self.hyp_dict["batch_size"]), False
        )

    def get_loss(self) -> dict:
        """Return train and test loss for all bottleneck sizes.

        Returns:
            bottleneck_loss (dict): train and test loss for different bottleneck sizes
        """
        return self.bottleneck_loss

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
