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
    if modelname == "celeba_inn":
        trainset, _ = load_data(True)
    else:
        trainset, _ = load_data()
    trainloader = get_loader(torch.utils.data.Subset(trainset, [1, 2, 3, 4]), 2, True)
    start_model = trainer.model
    start_loss = trainer.evaluate_inn(trainloader)
    trainer.train_inn(trainloader, "pytest")
    end_model = trainer.model
    end_loss = trainer.evaluate_inn(trainloader)
    assert start_model.parameters() != end_model.parameters()
    assert isinstance(start_loss, float)
    assert isinstance(end_loss, float)
    assert start_loss >= 0
    assert end_loss >= 0
    assert start_loss != end_loss
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


@pytest.mark.parametrize(
    "test_cases",
    [
        ("mnist_classic", load_mnist),
        ("mnist_classic1024", load_mnist),
        ("mnist_classicDeep1024", load_mnist),
        ("mnist_classic2048", load_mnist),
        ("cifar_classic", load_cifar),
        ("cifar_classic", load_cifar),
        ("celeba_classic", load_celeba),
    ],
)
def test_Trainer_classic(test_cases):
    modelname, load_data = test_cases
    hyp_dict = HYPERPARAMETER[modelname]
    hyp_dict["num_epoch"] = 3
    hyp_dict["milestones"] = [2]
    trainer = Trainer(5, modelname, hyp_dict)
    if modelname == "celeba_classic":
        trainset, _ = load_data(False)
    else:
        trainset, _ = load_data()
    trainloader = get_loader(torch.utils.data.Subset(trainset, [1, 2, 3, 4]), 2, True)
    start_model = trainer.model
    start_loss = trainer.evaluate_classic(trainloader)
    trainer.train_classic(trainloader, "pytest")
    end_model = trainer.model
    end_loss = trainer.evaluate_classic(trainloader)
    assert start_model.parameters() != end_model.parameters()
    assert isinstance(start_loss, float)
    assert isinstance(end_loss, float)
    assert start_loss >= 0
    assert end_loss >= 0
    assert start_loss != end_loss
    assert os.path.exists(os.path.join("models", "pytest", f"{modelname}.pt"))
    assert os.path.exists(os.path.join("logs", "pytest", f"{modelname}_train_rec.npy"))
    delete_file("models", f"{modelname}.pt", "pytest")
    delete_file("logs", f"{modelname}_train_rec.npy", "pytest")
    assert not os.path.exists(os.path.join("models", "pytest", f"{modelname}.pt"))
    assert not os.path.exists(
        os.path.join("logs", "pytest", f"{modelname}_train_rec.npy")
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
