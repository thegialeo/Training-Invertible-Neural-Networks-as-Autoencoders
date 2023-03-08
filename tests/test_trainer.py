import os

import pytest
import torch

from src.dataloader import get_loader, load_celeba, load_cifar, load_mnist
from src.filemanager import delete_file
from src.settings import HYPERPARAMETER
from src.trainer import Trainer


@pytest.mark.parametrize(
    "test_cases",
    [("mnist_inn", load_mnist), ("cifar_inn", load_cifar), ("celeba_inn", load_celeba)],
)
def test_Trainer_inn(test_cases):
    modelname, load_data = test_cases
    hyp_dict = HYPERPARAMETER[modelname]
    hyp_dict["num_epoch"] = 3
    hyp_dict["milestones"] = [2]
    trainer = Trainer(5, modelname, hyp_dict)
    start_model = trainer.model
    if modelname == "celeba_inn":
        trainset, _ = load_data(True)
    else:
        trainset, _ = load_data()
    trainloader = get_loader(torch.utils.data.Subset(trainset, [1, 2, 3, 4]), 2, True)
    trainer.train_inn(trainloader, "pytest")
    end_model = trainer.model
    assert start_model.parameters() != end_model.parameters()
    assert os.path.exists(os.path.join("models", "pytest", f"{modelname}.pt"))
    assert os.path.exists(
        os.path.join("logs", "pytest", f"{modelname}_train_total.npy")
    )
    assert os.path.exists(os.path.join("logs", "pytest", f"{modelname}_train_rec.npy"))
    assert os.path.exists(os.path.join("logs", "pytest", f"{modelname}_train_dist.npy"))
    assert os.path.exists(
        os.path.join("logs", "pytest", f"{modelname}_train_sparse.npy")
    )
    delete_file("models", f"{modelname}.pt", "pytest")
    delete_file("logs", f"{modelname}_train_total.npy", "pytest")
    delete_file("logs", f"{modelname}_train_rec.npy", "pytest")
    delete_file("logs", f"{modelname}_train_dist.npy", "pytest")
    delete_file("logs", f"{modelname}_train_sparse.npy", "pytest")
    assert not os.path.exists(os.path.join("models", "pytest", f"{modelname}.pt"))
    assert not os.path.exists(
        os.path.join("logs", "pytest", f"{modelname}_train_total.npy")
    )
    assert not os.path.exists(
        os.path.join("logs", "pytest", f"{modelname}_train_rec.npy")
    )
    assert not os.path.exists(
        os.path.join("logs", "pytest", f"{modelname}_train_dist.npy")
    )
    assert not os.path.exists(
        os.path.join("logs", "pytest", f"{modelname}_train_sparse.npy")
    )
    assert os.path.exists(os.path.join("models", "pytest"))
    assert os.path.exists(os.path.join("logs", "pytest"))
    if not os.listdir(os.path.join("models", "pytest")):
        os.rmdir(os.path.join("models", "pytest"))
        assert not os.path.exists(os.path.join("models", "pytest"))
    if not os.listdir(os.path.join("logs", "pytest")):
        os.rmdir(os.path.join("logs", "pytest"))
        assert not os.path.exists(os.path.join("logs", "pytest"))
    if not os.listdir("models"):
        os.rmdir(os.path.join("models"))
        assert not os.path.exists(os.path.join("models"))
    if not os.listdir("logs"):
        os.rmdir(os.path.join("logs"))
        assert not os.path.exists(os.path.join("logs"))
