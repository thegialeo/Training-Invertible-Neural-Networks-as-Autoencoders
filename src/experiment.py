"""Run experiments in the paper.

Classes:
    Experiment: Run experiments in the paper
"""
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
        trainloader: dataloader for trainset
        testloader: dataloader for testset

    Methods:
        run_inn_experiment() -> None: run bottleneck experiment for INN autoencoder
        run_classic_experiment() -> None: run bottleneck experiment for classic autoencoder
    """

    def __init__(self, modelname: str) -> None:
        """Initialize the experiment.

        Args:
            modelname (str): name of the model to run experiments
        """
        self.modelname = modelname
        self.hyp_dict = HYPERPARAMETER[modelname]
        self.bottleneck_loss = {
            "train": torch.zeros(len(cast(list[int], self.hyp_dict["lat_dim_lst"]))),
            "test": torch.zeros(len(cast(list[int], self.hyp_dict["lat_dim_lst"]))),
        }

        trainset, testset = DATASET[self.modelname]
        self.trainloader = get_loader(
            trainset, cast(int, self.hyp_dict["batch_size"]), True
        )
        self.testloader = get_loader(
            testset, cast(int, self.hyp_dict["batch_size"]), False
        )

    def run_inn_experiment(self) -> None:
        """Run INN autoencoder bottleneck experiment."""
        for idx, lat_dim in enumerate(cast(list[int], self.hyp_dict["lat_dim_lst"])):
            print("\n")
            print(f"Start Training with latent dimension: {lat_dim}")
            print("\n")

            trainer = Trainer(lat_dim, self.modelname, self.hyp_dict)
            trainer.train_inn(self.trainloader)

            train_loss = trainer.evaluate_inn(self.trainloader)
            test_loss = trainer.evaluate_inn(self.testloader)
            self.bottleneck_loss["train"][idx] = train_loss
            self.bottleneck_loss["test"][idx] = test_loss

            trainer.plot_inn(self.testloader, 100, 10, "latent dimension {lat_dim}")

    def run_classic_experiment(self) -> None:
        """Run classic autoencoder bottleneck experiment."""
        for idx, lat_dim in enumerate(cast(list[int], self.hyp_dict["lat_dim_lst"])):
            print("\n")
            print(f"Start Training with latent dimension: {lat_dim}")
            print("\n")

            trainer = Trainer(lat_dim, self.modelname, self.hyp_dict)
            trainer.train_classic(self.trainloader)

            train_loss = trainer.evaluate_classic(self.trainloader)
            test_loss = trainer.evaluate_classic(self.testloader)
            self.bottleneck_loss["train"][idx] = train_loss
            self.bottleneck_loss["test"][idx] = test_loss

            trainer.plot_classic(self.testloader, 100, 10, "latent dimension {lat_dim}")
