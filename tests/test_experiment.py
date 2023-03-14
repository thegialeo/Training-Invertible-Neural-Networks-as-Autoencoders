import pytest
import torch

from src.dataloader import DATASET, get_loader
from src.experiment import Experiment


@pytest.mark.parametrize("modelname", ["mnist_inn", "cifar_inn", "celeba_inn"])
def test_experiment_inn(modelname):
    experiment = Experiment(modelname)
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
    experiment.run_inn_experiment()
    assert torch.sum(experiment.bottleneck_loss["train"] > 0) == 2
    assert (
        experiment.bottleneck_loss["train"][0] != experiment.bottleneck_loss["train"][1]
    )
    assert torch.sum(experiment.bottleneck_loss["test"] > 0) == 2
    assert (
        experiment.bottleneck_loss["test"][0] != experiment.bottleneck_loss["test"][1]
    )


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
    experiment = Experiment(modelname)
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
    experiment.run_classic_experiment()
    assert torch.sum(experiment.bottleneck_loss["train"] > 0) == 2
    assert (
        experiment.bottleneck_loss["train"][0] != experiment.bottleneck_loss["train"][1]
    )
    assert torch.sum(experiment.bottleneck_loss["test"] > 0) == 2
    assert (
        experiment.bottleneck_loss["test"][0] != experiment.bottleneck_loss["test"][1]
    )
