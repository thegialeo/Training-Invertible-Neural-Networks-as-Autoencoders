import os

import pytest
import torch

from src.dataloader import DATASET, get_loader
from src.experiment import Experiment
from src.filemanager import delete_file


@pytest.mark.parametrize("modelname", ["mnist_inn", "cifar_inn", "celeba_inn"])
def test_experiment_inn(modelname):
    experiment = Experiment(modelname, "pytest")
    experiment.hyp_dict["num_epoch"] = 3
    experiment.hyp_dict["milestones"] = [2]
    experiment.hyp_dict["lat_dim_lst"] = [3, 5]
    trainset, testset = DATASET[modelname]
    experiment.trainloader = get_loader(
        torch.utils.data.Subset(trainset, [1, 2, 3, 4, 5, 6, 7, 8]), 4, True
    )
    experiment.testloader = get_loader(
        torch.utils.data.Subset(testset, [1, 2, 3, 4, 5, 6, 7, 8]), 4, False
    )
    experiment.run_experiment()
    lat_dim_lst = experiment.get_lat_dim_lst()
    train_loss = experiment.get_loss("train")
    test_loss = experiment.get_loss("test")
    assert isinstance(lat_dim_lst, list)
    assert isinstance(train_loss, torch.Tensor)
    assert isinstance(test_loss, torch.Tensor)
    assert lat_dim_lst == [3, 5]
    assert torch.equal(train_loss, experiment.bottleneck_loss["train"])
    assert torch.equal(test_loss, experiment.bottleneck_loss["test"])
    assert torch.sum(train_loss > 0) == 2
    assert train_loss[0] != train_loss[1]
    assert torch.sum(test_loss > 0) == 2
    assert test_loss[0] != test_loss[1]
    assert os.path.exists(
        os.path.join("models", "pytest", "lat_dim_3", f"{modelname}.pt")
    )
    assert os.path.exists(
        os.path.join("models", "pytest", "lat_dim_5", f"{modelname}.pt")
    )
    assert os.path.exists(
        os.path.join("logs", "pytest", "lat_dim_3", f"{modelname}_train_total.npy")
    )
    assert os.path.exists(
        os.path.join("logs", "pytest", "lat_dim_3", f"{modelname}_train_rec.npy")
    )
    assert os.path.exists(
        os.path.join("logs", "pytest", "lat_dim_3", f"{modelname}_train_dist.npy")
    )
    assert os.path.exists(
        os.path.join("logs", "pytest", "lat_dim_3", f"{modelname}_train_sparse.npy")
    )
    assert os.path.exists(
        os.path.join("logs", "pytest", "lat_dim_5", f"{modelname}_train_total.npy")
    )
    assert os.path.exists(
        os.path.join("logs", "pytest", "lat_dim_5", f"{modelname}_train_rec.npy")
    )
    assert os.path.exists(
        os.path.join("logs", "pytest", "lat_dim_5", f"{modelname}_train_dist.npy")
    )
    assert os.path.exists(
        os.path.join("logs", "pytest", "lat_dim_5", f"{modelname}_train_sparse.npy")
    )
    assert os.path.exists(
        os.path.join("plots", "pytest", "lat_dim_3", f"{modelname}_original.png")
    )
    assert os.path.exists(
        os.path.join("plots", "pytest", "lat_dim_3", f"{modelname}_reconstructed.png")
    )
    assert os.path.exists(
        os.path.join("plots", "pytest", "lat_dim_3", f"{modelname}_difference.png")
    )
    assert os.path.exists(
        os.path.join("plots", "pytest", "lat_dim_5", f"{modelname}_original.png")
    )
    assert os.path.exists(
        os.path.join("plots", "pytest", "lat_dim_5", f"{modelname}_reconstructed.png")
    )
    assert os.path.exists(
        os.path.join("plots", "pytest", "lat_dim_5", f"{modelname}_difference.png")
    )
    delete_file("models", f"{modelname}.pt", os.path.join("pytest", "lat_dim_3"))
    delete_file("models", f"{modelname}.pt", os.path.join("pytest", "lat_dim_5"))
    delete_file(
        "logs", f"{modelname}_train_total.npy", os.path.join("pytest", "lat_dim_3")
    )
    delete_file(
        "logs", f"{modelname}_train_rec.npy", os.path.join("pytest", "lat_dim_3")
    )
    delete_file(
        "logs", f"{modelname}_train_dist.npy", os.path.join("pytest", "lat_dim_3")
    )
    delete_file(
        "logs", f"{modelname}_train_sparse.npy", os.path.join("pytest", "lat_dim_3")
    )
    delete_file(
        "logs", f"{modelname}_train_total.npy", os.path.join("pytest", "lat_dim_5")
    )
    delete_file(
        "logs", f"{modelname}_train_rec.npy", os.path.join("pytest", "lat_dim_5")
    )
    delete_file(
        "logs", f"{modelname}_train_dist.npy", os.path.join("pytest", "lat_dim_5")
    )
    delete_file(
        "logs", f"{modelname}_train_sparse.npy", os.path.join("pytest", "lat_dim_5")
    )
    delete_file(
        "plots", f"{modelname}_original.png", os.path.join("pytest", "lat_dim_3")
    )
    delete_file(
        "plots", f"{modelname}_reconstructed.png", os.path.join("pytest", "lat_dim_3")
    )
    delete_file(
        "plots", f"{modelname}_difference.png", os.path.join("pytest", "lat_dim_3")
    )
    delete_file(
        "plots", f"{modelname}_original.png", os.path.join("pytest", "lat_dim_5")
    )
    delete_file(
        "plots", f"{modelname}_reconstructed.png", os.path.join("pytest", "lat_dim_5")
    )
    delete_file(
        "plots", f"{modelname}_difference.png", os.path.join("pytest", "lat_dim_5")
    )
    assert not os.path.exists(
        os.path.join("models", "pytest", "lat_dim_3", f"{modelname}.pt")
    )
    assert not os.path.exists(
        os.path.join("models", "pytest", "lat_dim_5", f"{modelname}.pt")
    )
    assert not os.path.exists(
        os.path.join("logs", "pytest", "lat_dim_3", f"{modelname}_train_total.npy")
    )
    assert not os.path.exists(
        os.path.join("logs", "pytest", "lat_dim_3", f"{modelname}_train_rec.npy")
    )
    assert not os.path.exists(
        os.path.join("logs", "pytest", "lat_dim_3", f"{modelname}_train_dist.npy")
    )
    assert not os.path.exists(
        os.path.join("logs", "pytest", "lat_dim_3", f"{modelname}_train_sparse.npy")
    )
    assert not os.path.exists(
        os.path.join("logs", "pytest", "lat_dim_5", f"{modelname}_train_total.npy")
    )
    assert not os.path.exists(
        os.path.join("logs", "pytest", "lat_dim_5", f"{modelname}_train_rec.npy")
    )
    assert not os.path.exists(
        os.path.join("logs", "pytest", "lat_dim_5", f"{modelname}_train_dist.npy")
    )
    assert not os.path.exists(
        os.path.join("logs", "pytest", "lat_dim_5", f"{modelname}_train_sparse.npy")
    )
    assert not os.path.exists(
        os.path.join("plots", "pytest", "lat_dim_3", f"{modelname}_original.png")
    )
    assert not os.path.exists(
        os.path.join("plots", "pytest", "lat_dim_3", f"{modelname}_reconstructed.png")
    )
    assert not os.path.exists(
        os.path.join("plots", "pytest", "lat_dim_3", f"{modelname}_difference.png")
    )
    assert not os.path.exists(
        os.path.join("plots", "pytest", "lat_dim_5", f"{modelname}_original.png")
    )
    assert not os.path.exists(
        os.path.join("plots", "pytest", "lat_dim_5", f"{modelname}_reconstructed.png")
    )
    assert not os.path.exists(
        os.path.join("plots", "pytest", "lat_dim_5", f"{modelname}_difference.png")
    )
    assert os.path.exists(os.path.join("models", "pytest", "lat_dim_3"))
    assert os.path.exists(os.path.join("models", "pytest", "lat_dim_5"))
    assert os.path.exists(os.path.join("logs", "pytest", "lat_dim_3"))
    assert os.path.exists(os.path.join("logs", "pytest", "lat_dim_5"))
    assert os.path.exists(os.path.join("plots", "pytest", "lat_dim_3"))
    assert os.path.exists(os.path.join("plots", "pytest", "lat_dim_5"))
    if not os.listdir(os.path.join("models", "pytest", "lat_dim_3")):
        os.rmdir(os.path.join("models", "pytest", "lat_dim_3"))
    if not os.listdir(os.path.join("models", "pytest", "lat_dim_5")):
        os.rmdir(os.path.join("models", "pytest", "lat_dim_5"))
    if not os.listdir(os.path.join("logs", "pytest", "lat_dim_3")):
        os.rmdir(os.path.join("logs", "pytest", "lat_dim_3"))
    if not os.listdir(os.path.join("logs", "pytest", "lat_dim_5")):
        os.rmdir(os.path.join("logs", "pytest", "lat_dim_5"))
    if not os.listdir(os.path.join("plots", "pytest", "lat_dim_3")):
        os.rmdir(os.path.join("plots", "pytest", "lat_dim_3"))
    if not os.listdir(os.path.join("plots", "pytest", "lat_dim_5")):
        os.rmdir(os.path.join("plots", "pytest", "lat_dim_5"))
    assert not os.path.exists(os.path.join("models", "pytest", "lat_dim_3"))
    assert not os.path.exists(os.path.join("models", "pytest", "lat_dim_5"))
    assert not os.path.exists(os.path.join("logs", "pytest", "lat_dim_3"))
    assert not os.path.exists(os.path.join("logs", "pytest", "lat_dim_5"))
    assert not os.path.exists(os.path.join("plots", "pytest", "lat_dim_3"))
    assert not os.path.exists(os.path.join("plots", "pytest", "lat_dim_5"))
    assert os.path.exists(os.path.join("models", "pytest"))
    assert os.path.exists(os.path.join("logs", "pytest"))
    assert os.path.exists(os.path.join("plots", "pytest"))
    if not os.listdir(os.path.join("models", "pytest")):
        os.rmdir(os.path.join("models", "pytest"))
    if not os.listdir(os.path.join("logs", "pytest")):
        os.rmdir(os.path.join("logs", "pytest"))
    if not os.listdir(os.path.join("plots", "pytest")):
        os.rmdir(os.path.join("plots", "pytest"))
    assert not os.path.exists(os.path.join("models", "pytest"))
    assert not os.path.exists(os.path.join("logs", "pytest"))
    assert not os.path.exists(os.path.join("plots", "pytest"))
    if not os.listdir("models"):
        os.rmdir(os.path.join("models"))
        assert not os.path.exists(os.path.join("models"))
    if not os.listdir("logs"):
        os.rmdir(os.path.join("logs"))
        assert not os.path.exists(os.path.join("logs"))
    if not os.listdir("plots"):
        os.rmdir(os.path.join("plots"))
        assert not os.path.exists(os.path.join("plots"))


@pytest.mark.parametrize(
    "modelname",
    [
        "mnist_classic",
        "mnist_classic1024",
        "mnist_classicDeep1024",
        "mnist_classic2048",
        "cifar_classic",
        "celeba_classic",
    ],
)
def test_experiment_classic(modelname):
    experiment = Experiment(modelname, "pytest")
    experiment.hyp_dict["num_epoch"] = 3
    experiment.hyp_dict["milestones"] = [2]
    experiment.hyp_dict["lat_dim_lst"] = [3, 5]
    trainset, testset = DATASET[modelname]
    experiment.trainloader = get_loader(
        torch.utils.data.Subset(trainset, [1, 2, 3, 4, 5, 6, 7, 8]), 4, True
    )
    experiment.testloader = get_loader(
        torch.utils.data.Subset(testset, [1, 2, 3, 4, 5, 6, 7, 8]), 4, False
    )
    experiment.run_experiment()
    lat_dim_lst = experiment.get_lat_dim_lst()
    train_loss = experiment.get_loss("train")
    test_loss = experiment.get_loss("test")
    assert isinstance(lat_dim_lst, list)
    assert isinstance(train_loss, torch.Tensor)
    assert isinstance(test_loss, torch.Tensor)
    assert lat_dim_lst == [3, 5]
    assert torch.equal(train_loss, experiment.bottleneck_loss["train"])
    assert torch.equal(test_loss, experiment.bottleneck_loss["test"])
    assert torch.sum(train_loss > 0) == 2
    assert train_loss[0] != train_loss[1]
    assert torch.sum(test_loss > 0) == 2
    assert test_loss[0] != test_loss[1]
    assert os.path.exists(
        os.path.join("models", "pytest", "lat_dim_3", f"{modelname}.pt")
    )
    assert os.path.exists(
        os.path.join("models", "pytest", "lat_dim_5", f"{modelname}.pt")
    )
    assert os.path.exists(
        os.path.join("logs", "pytest", "lat_dim_3", f"{modelname}_train_rec.npy")
    )
    assert os.path.exists(
        os.path.join("logs", "pytest", "lat_dim_5", f"{modelname}_train_rec.npy")
    )
    assert os.path.exists(
        os.path.join("plots", "pytest", "lat_dim_3", f"{modelname}_original.png")
    )
    assert os.path.exists(
        os.path.join("plots", "pytest", "lat_dim_3", f"{modelname}_reconstructed.png")
    )
    assert os.path.exists(
        os.path.join("plots", "pytest", "lat_dim_3", f"{modelname}_difference.png")
    )
    assert os.path.exists(
        os.path.join("plots", "pytest", "lat_dim_5", f"{modelname}_original.png")
    )
    assert os.path.exists(
        os.path.join("plots", "pytest", "lat_dim_5", f"{modelname}_reconstructed.png")
    )
    assert os.path.exists(
        os.path.join("plots", "pytest", "lat_dim_5", f"{modelname}_difference.png")
    )
    delete_file("models", f"{modelname}.pt", os.path.join("pytest", "lat_dim_3"))
    delete_file("models", f"{modelname}.pt", os.path.join("pytest", "lat_dim_5"))
    delete_file(
        "logs", f"{modelname}_train_rec.npy", os.path.join("pytest", "lat_dim_3")
    )
    delete_file(
        "logs", f"{modelname}_train_rec.npy", os.path.join("pytest", "lat_dim_5")
    )
    delete_file(
        "plots", f"{modelname}_original.png", os.path.join("pytest", "lat_dim_3")
    )
    delete_file(
        "plots", f"{modelname}_reconstructed.png", os.path.join("pytest", "lat_dim_3")
    )
    delete_file(
        "plots", f"{modelname}_difference.png", os.path.join("pytest", "lat_dim_3")
    )
    delete_file(
        "plots", f"{modelname}_original.png", os.path.join("pytest", "lat_dim_5")
    )
    delete_file(
        "plots", f"{modelname}_reconstructed.png", os.path.join("pytest", "lat_dim_5")
    )
    delete_file(
        "plots", f"{modelname}_difference.png", os.path.join("pytest", "lat_dim_5")
    )
    assert not os.path.exists(
        os.path.join("models", "pytest", "lat_dim_3", f"{modelname}.pt")
    )
    assert not os.path.exists(
        os.path.join("models", "pytest", "lat_dim_5", f"{modelname}.pt")
    )
    assert not os.path.exists(
        os.path.join("logs", "pytest", "lat_dim_3", f"{modelname}_train_rec.npy")
    )
    assert not os.path.exists(
        os.path.join("logs", "pytest", "lat_dim_5", f"{modelname}_train_rec.npy")
    )
    assert not os.path.exists(
        os.path.join("plots", "pytest", "lat_dim_3", f"{modelname}_original.png")
    )
    assert not os.path.exists(
        os.path.join("plots", "pytest", "lat_dim_3", f"{modelname}_reconstructed.png")
    )
    assert not os.path.exists(
        os.path.join("plots", "pytest", "lat_dim_3", f"{modelname}_difference.png")
    )
    assert not os.path.exists(
        os.path.join("plots", "pytest", "lat_dim_5", f"{modelname}_original.png")
    )
    assert not os.path.exists(
        os.path.join("plots", "pytest", "lat_dim_5", f"{modelname}_reconstructed.png")
    )
    assert not os.path.exists(
        os.path.join("plots", "pytest", "lat_dim_5", f"{modelname}_difference.png")
    )
    assert os.path.exists(os.path.join("models", "pytest", "lat_dim_3"))
    assert os.path.exists(os.path.join("models", "pytest", "lat_dim_5"))
    assert os.path.exists(os.path.join("logs", "pytest", "lat_dim_3"))
    assert os.path.exists(os.path.join("logs", "pytest", "lat_dim_5"))
    assert os.path.exists(os.path.join("plots", "pytest", "lat_dim_3"))
    assert os.path.exists(os.path.join("plots", "pytest", "lat_dim_5"))
    if not os.listdir(os.path.join("models", "pytest", "lat_dim_3")):
        os.rmdir(os.path.join("models", "pytest", "lat_dim_3"))
    if not os.listdir(os.path.join("models", "pytest", "lat_dim_5")):
        os.rmdir(os.path.join("models", "pytest", "lat_dim_5"))
    if not os.listdir(os.path.join("logs", "pytest", "lat_dim_3")):
        os.rmdir(os.path.join("logs", "pytest", "lat_dim_3"))
    if not os.listdir(os.path.join("logs", "pytest", "lat_dim_5")):
        os.rmdir(os.path.join("logs", "pytest", "lat_dim_5"))
    if not os.listdir(os.path.join("plots", "pytest", "lat_dim_3")):
        os.rmdir(os.path.join("plots", "pytest", "lat_dim_3"))
    if not os.listdir(os.path.join("plots", "pytest", "lat_dim_5")):
        os.rmdir(os.path.join("plots", "pytest", "lat_dim_5"))
    assert not os.path.exists(os.path.join("models", "pytest", "lat_dim_3"))
    assert not os.path.exists(os.path.join("models", "pytest", "lat_dim_5"))
    assert not os.path.exists(os.path.join("logs", "pytest", "lat_dim_3"))
    assert not os.path.exists(os.path.join("logs", "pytest", "lat_dim_5"))
    assert not os.path.exists(os.path.join("plots", "pytest", "lat_dim_3"))
    assert not os.path.exists(os.path.join("plots", "pytest", "lat_dim_5"))
    assert os.path.exists(os.path.join("models", "pytest"))
    assert os.path.exists(os.path.join("logs", "pytest"))
    assert os.path.exists(os.path.join("plots", "pytest"))
    if not os.listdir(os.path.join("models", "pytest")):
        os.rmdir(os.path.join("models", "pytest"))
    if not os.listdir(os.path.join("logs", "pytest")):
        os.rmdir(os.path.join("logs", "pytest"))
    if not os.listdir(os.path.join("plots", "pytest")):
        os.rmdir(os.path.join("plots", "pytest"))
    assert not os.path.exists(os.path.join("models", "pytest"))
    assert not os.path.exists(os.path.join("logs", "pytest"))
    assert not os.path.exists(os.path.join("plots", "pytest"))
    if not os.listdir("models"):
        os.rmdir(os.path.join("models"))
        assert not os.path.exists(os.path.join("models"))
    if not os.listdir("logs"):
        os.rmdir(os.path.join("logs"))
        assert not os.path.exists(os.path.join("logs"))
    if not os.listdir("plots"):
        os.rmdir(os.path.join("plots"))
        assert not os.path.exists(os.path.join("plots"))
