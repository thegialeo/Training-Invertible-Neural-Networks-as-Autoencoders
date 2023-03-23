import os

import pytest
import torch

from src.experiment import Experiment, experiment_wrapper
from src.filemanager import delete_file


@pytest.mark.parametrize(
    "modelname", ["pytest_mnist_inn", "pytest_cifar_inn", "pytest_celeba_inn"]
)
def test_experiment_inn(modelname):
    experiment = Experiment(modelname, "pytest")
    experiment.run_experiment()
    lat_dim_lst = experiment.get_lat_dim_lst()
    train_loss = experiment.get_loss("train")
    test_loss = experiment.get_loss("test")
    num_params = experiment.get_model_param_count()
    assert isinstance(lat_dim_lst, list)
    assert isinstance(train_loss, torch.Tensor)
    assert isinstance(test_loss, torch.Tensor)
    assert isinstance(num_params, list)
    assert isinstance(num_params[0], int)
    assert isinstance(num_params[1], int)
    assert lat_dim_lst == [3, 5]
    assert torch.equal(train_loss, experiment.bottleneck_loss["train"])
    assert torch.equal(test_loss, experiment.bottleneck_loss["test"])
    assert train_loss.size() == (2,)
    assert train_loss[0] != train_loss[1]
    assert test_loss.size() == (2,)
    assert test_loss[0] != test_loss[1]
    assert len(num_params) == 2
    assert num_params[0] > 0
    assert num_params[1] > 0
    assert num_params[0] == num_params[1]
    assert os.path.exists(
        os.path.join("models", "pytest", "lat_dim_3", f"{modelname}.pt")
    )
    assert os.path.exists(
        os.path.join("models", "pytest", "lat_dim_5", f"{modelname}.pt")
    )
    assert os.path.exists(
        os.path.join("logs", "pytest", f"{modelname}_bottleneck_train_loss.npy")
    )
    assert os.path.exists(
        os.path.join("logs", "pytest", f"{modelname}_bottleneck_test_loss.npy")
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
    delete_file("logs", f"{modelname}_bottleneck_train_loss.npy", "pytest")
    delete_file("logs", f"{modelname}_bottleneck_test_loss.npy", "pytest")
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
        os.path.join("logs", "pytest", f"{modelname}_bottleneck_train_loss.npy")
    )
    assert not os.path.exists(
        os.path.join("logs", "pytest", f"{modelname}_bottleneck_test_loss.npy")
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
    os.rmdir(os.path.join("models", "pytest", "lat_dim_3"))
    os.rmdir(os.path.join("models", "pytest", "lat_dim_5"))
    os.rmdir(os.path.join("logs", "pytest", "lat_dim_3"))
    os.rmdir(os.path.join("logs", "pytest", "lat_dim_5"))
    os.rmdir(os.path.join("plots", "pytest", "lat_dim_3"))
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
    os.rmdir(os.path.join("models", "pytest"))
    os.rmdir(os.path.join("logs", "pytest"))
    os.rmdir(os.path.join("plots", "pytest"))
    assert not os.path.exists(os.path.join("models", "pytest"))
    assert not os.path.exists(os.path.join("logs", "pytest"))
    assert not os.path.exists(os.path.join("plots", "pytest"))
    assert os.path.exists("models")
    assert os.path.exists("logs")
    assert os.path.exists("plots")
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
        "pytest_mnist_classic",
        "pytest_mnist_classic1024",
        "pytest_mnist_classicDeep1024",
        "pytest_mnist_classic2048",
        "pytest_cifar_classic",
        "pytest_celeba_classic",
    ],
)
def test_experiment_classic(modelname):
    experiment = Experiment(modelname, "pytest")
    experiment.run_experiment()
    lat_dim_lst = experiment.get_lat_dim_lst()
    train_loss = experiment.get_loss("train")
    test_loss = experiment.get_loss("test")
    num_params = experiment.get_model_param_count()
    assert isinstance(lat_dim_lst, list)
    assert isinstance(train_loss, torch.Tensor)
    assert isinstance(test_loss, torch.Tensor)
    assert isinstance(num_params, list)
    assert isinstance(num_params[0], int)
    assert isinstance(num_params[1], int)
    assert lat_dim_lst == [3, 5]
    assert torch.equal(train_loss, experiment.bottleneck_loss["train"])
    assert torch.equal(test_loss, experiment.bottleneck_loss["test"])
    assert train_loss.size() == (2,)
    assert train_loss[0] != train_loss[1]
    assert test_loss.size() == (2,)
    assert test_loss[0] != test_loss[1]
    assert len(num_params) == 2
    assert num_params[0] > 0
    assert num_params[1] > 0
    assert num_params[0] < num_params[1]
    assert os.path.exists(
        os.path.join("models", "pytest", "lat_dim_3", f"{modelname}.pt")
    )
    assert os.path.exists(
        os.path.join("models", "pytest", "lat_dim_5", f"{modelname}.pt")
    )
    assert os.path.exists(
        os.path.join("logs", "pytest", f"{modelname}_bottleneck_train_loss.npy")
    )
    assert os.path.exists(
        os.path.join("logs", "pytest", f"{modelname}_bottleneck_test_loss.npy")
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
    delete_file("logs", f"{modelname}_bottleneck_train_loss.npy", "pytest")
    delete_file("logs", f"{modelname}_bottleneck_test_loss.npy", "pytest")
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
        os.path.join("logs", "pytest", f"{modelname}_bottleneck_train_loss.npy")
    )
    assert not os.path.exists(
        os.path.join("logs", "pytest", f"{modelname}_bottleneck_test_loss.npy")
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
    os.rmdir(os.path.join("models", "pytest", "lat_dim_3"))
    os.rmdir(os.path.join("models", "pytest", "lat_dim_5"))
    os.rmdir(os.path.join("logs", "pytest", "lat_dim_3"))
    os.rmdir(os.path.join("logs", "pytest", "lat_dim_5"))
    os.rmdir(os.path.join("plots", "pytest", "lat_dim_3"))
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
    os.rmdir(os.path.join("models", "pytest"))
    os.rmdir(os.path.join("logs", "pytest"))
    os.rmdir(os.path.join("plots", "pytest"))
    assert not os.path.exists(os.path.join("models", "pytest"))
    assert not os.path.exists(os.path.join("logs", "pytest"))
    assert not os.path.exists(os.path.join("plots", "pytest"))
    assert os.path.exists("models")
    assert os.path.exists("logs")
    assert os.path.exists("plots")
    if not os.listdir("models"):
        os.rmdir(os.path.join("models"))
        assert not os.path.exists(os.path.join("models"))
    if not os.listdir("logs"):
        os.rmdir(os.path.join("logs"))
        assert not os.path.exists(os.path.join("logs"))
    if not os.listdir("plots"):
        os.rmdir(os.path.join("plots"))
        assert not os.path.exists(os.path.join("plots"))


def test_experiment_wrapper():
    model_lst = [
        "pytest_mnist_inn",
        "pytest_cifar_inn",
        "pytest_celeba_inn",
        "pytest_mnist_classic",
        "pytest_mnist_classic1024",
        "pytest_mnist_classicDeep1024",
        "pytest_mnist_classic2048",
        "pytest_cifar_classic",
        "pytest_celeba_classic",
    ]
    lat_dim_lst, train_loss_lst, test_loss_lst, num_param_lst = experiment_wrapper(
        model_lst, "pytest"
    )
    assert isinstance(lat_dim_lst, list)
    assert isinstance(train_loss_lst, list)
    assert isinstance(test_loss_lst, list)
    assert isinstance(num_param_lst, list)
    assert len(lat_dim_lst) == len(model_lst)
    assert len(train_loss_lst) == len(model_lst)
    assert len(test_loss_lst) == len(model_lst)
    assert len(num_param_lst) == len(model_lst)
    for lat_dim in lat_dim_lst:
        assert lat_dim == [3, 5]
    for train_loss, test_loss in zip(train_loss_lst, test_loss_lst):
        assert isinstance(train_loss, torch.Tensor)
        assert isinstance(test_loss, torch.Tensor)
        assert train_loss.size() == (2,)
        assert train_loss[0] != train_loss[1]
        assert test_loss.size() == (2,)
        assert test_loss[0] != test_loss[1]
    for num_param in num_param_lst:
        assert isinstance(num_param, list)
        assert isinstance(num_param[0], int)
        assert isinstance(num_param[1], int)
        assert num_param[0] > 0
        assert num_param[1] > 0
        assert num_param[0] <= num_param[1]

    assert os.path.exists(os.path.join("plots", "pytest", "train_bottleneck_loss.png"))
    assert os.path.exists(os.path.join("plots", "pytest", "test_bottleneck_loss.png"))
    assert os.path.exists(os.path.join("plots", "pytest", "num_param_comparison.png"))
    delete_file("plots", "train_bottleneck_loss.png", "pytest")
    delete_file("plots", "test_bottleneck_loss.png", "pytest")
    delete_file("plots", "num_param_comparison.png", "pytest")
    assert not os.path.exists(
        os.path.join("plots", "pytest", "train_bottleneck_loss.png")
    )
    assert not os.path.exists(
        os.path.join("plots", "pytest", "test_bottleneck_loss.png")
    )
    assert not os.path.exists(
        os.path.join("plots", "pytest", "num_param_comparison.png")
    )

    for inn_modelname in model_lst[:3]:
        assert os.path.exists(
            os.path.join(
                "logs", "pytest", "lat_dim_3", f"{inn_modelname}_train_total.npy"
            )
        )
        assert os.path.exists(
            os.path.join(
                "logs", "pytest", "lat_dim_3", f"{inn_modelname}_train_dist.npy"
            )
        )
        assert os.path.exists(
            os.path.join(
                "logs", "pytest", "lat_dim_3", f"{inn_modelname}_train_sparse.npy"
            )
        )
        assert os.path.exists(
            os.path.join(
                "logs", "pytest", "lat_dim_5", f"{inn_modelname}_train_total.npy"
            )
        )
        assert os.path.exists(
            os.path.join(
                "logs", "pytest", "lat_dim_5", f"{inn_modelname}_train_dist.npy"
            )
        )
        assert os.path.exists(
            os.path.join(
                "logs", "pytest", "lat_dim_5", f"{inn_modelname}_train_sparse.npy"
            )
        )
        delete_file(
            "logs",
            f"{inn_modelname}_train_total.npy",
            os.path.join("pytest", "lat_dim_3"),
        )
        delete_file(
            "logs",
            f"{inn_modelname}_train_dist.npy",
            os.path.join("pytest", "lat_dim_3"),
        )
        delete_file(
            "logs",
            f"{inn_modelname}_train_sparse.npy",
            os.path.join("pytest", "lat_dim_3"),
        )
        delete_file(
            "logs",
            f"{inn_modelname}_train_total.npy",
            os.path.join("pytest", "lat_dim_5"),
        )
        delete_file(
            "logs",
            f"{inn_modelname}_train_dist.npy",
            os.path.join("pytest", "lat_dim_5"),
        )
        delete_file(
            "logs",
            f"{inn_modelname}_train_sparse.npy",
            os.path.join("pytest", "lat_dim_5"),
        )
        assert not os.path.exists(
            os.path.join(
                "logs", "pytest", "lat_dim_3", f"{inn_modelname}_train_total.npy"
            )
        )
        assert not os.path.exists(
            os.path.join(
                "logs", "pytest", "lat_dim_3", f"{inn_modelname}_train_dist.npy"
            )
        )
        assert not os.path.exists(
            os.path.join(
                "logs", "pytest", "lat_dim_3", f"{inn_modelname}_train_sparse.npy"
            )
        )
        assert not os.path.exists(
            os.path.join(
                "logs", "pytest", "lat_dim_5", f"{inn_modelname}_train_total.npy"
            )
        )
        assert not os.path.exists(
            os.path.join(
                "logs", "pytest", "lat_dim_5", f"{inn_modelname}_train_dist.npy"
            )
        )
        assert not os.path.exists(
            os.path.join(
                "logs", "pytest", "lat_dim_5", f"{inn_modelname}_train_sparse.npy"
            )
        )

    for modelname in model_lst:
        assert os.path.exists(
            os.path.join("models", "pytest", "lat_dim_3", f"{modelname}.pt")
        )
        assert os.path.exists(
            os.path.join("models", "pytest", "lat_dim_5", f"{modelname}.pt")
        )
        assert os.path.exists(
            os.path.join("logs", "pytest", f"{modelname}_bottleneck_train_loss.npy")
        )
        assert os.path.exists(
            os.path.join("logs", "pytest", f"{modelname}_bottleneck_test_loss.npy")
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
            os.path.join(
                "plots", "pytest", "lat_dim_3", f"{modelname}_reconstructed.png"
            )
        )
        assert os.path.exists(
            os.path.join("plots", "pytest", "lat_dim_3", f"{modelname}_difference.png")
        )
        assert os.path.exists(
            os.path.join("plots", "pytest", "lat_dim_5", f"{modelname}_original.png")
        )
        assert os.path.exists(
            os.path.join(
                "plots", "pytest", "lat_dim_5", f"{modelname}_reconstructed.png"
            )
        )
        assert os.path.exists(
            os.path.join("plots", "pytest", "lat_dim_5", f"{modelname}_difference.png")
        )
        delete_file("models", f"{modelname}.pt", os.path.join("pytest", "lat_dim_3"))
        delete_file("models", f"{modelname}.pt", os.path.join("pytest", "lat_dim_5"))
        delete_file("logs", f"{modelname}_bottleneck_train_loss.npy", "pytest")
        delete_file("logs", f"{modelname}_bottleneck_test_loss.npy", "pytest")
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
            "plots",
            f"{modelname}_reconstructed.png",
            os.path.join("pytest", "lat_dim_3"),
        )
        delete_file(
            "plots", f"{modelname}_difference.png", os.path.join("pytest", "lat_dim_3")
        )
        delete_file(
            "plots", f"{modelname}_original.png", os.path.join("pytest", "lat_dim_5")
        )
        delete_file(
            "plots",
            f"{modelname}_reconstructed.png",
            os.path.join("pytest", "lat_dim_5"),
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
            os.path.join("logs", "pytest", f"{modelname}_bottleneck_train_loss.npy")
        )
        assert not os.path.exists(
            os.path.join("logs", "pytest", f"{modelname}_bottleneck_test_loss.npy")
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
            os.path.join(
                "plots", "pytest", "lat_dim_3", f"{modelname}_reconstructed.png"
            )
        )
        assert not os.path.exists(
            os.path.join("plots", "pytest", "lat_dim_3", f"{modelname}_difference.png")
        )
        assert not os.path.exists(
            os.path.join("plots", "pytest", "lat_dim_5", f"{modelname}_original.png")
        )
        assert not os.path.exists(
            os.path.join(
                "plots", "pytest", "lat_dim_5", f"{modelname}_reconstructed.png"
            )
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
    os.rmdir(os.path.join("models", "pytest", "lat_dim_3"))
    os.rmdir(os.path.join("models", "pytest", "lat_dim_5"))
    os.rmdir(os.path.join("logs", "pytest", "lat_dim_3"))
    os.rmdir(os.path.join("logs", "pytest", "lat_dim_5"))
    os.rmdir(os.path.join("plots", "pytest", "lat_dim_3"))
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
    os.rmdir(os.path.join("models", "pytest"))
    os.rmdir(os.path.join("logs", "pytest"))
    os.rmdir(os.path.join("plots", "pytest"))
    assert not os.path.exists(os.path.join("models", "pytest"))
    assert not os.path.exists(os.path.join("logs", "pytest"))
    assert not os.path.exists(os.path.join("plots", "pytest"))
    assert os.path.exists("models")
    assert os.path.exists("logs")
    assert os.path.exists("plots")
    if not os.listdir("models"):
        os.rmdir(os.path.join("models"))
        assert not os.path.exists(os.path.join("models"))
    if not os.listdir("logs"):
        os.rmdir(os.path.join("logs"))
        assert not os.path.exists(os.path.join("logs"))
    if not os.listdir("plots"):
        os.rmdir(os.path.join("plots"))
        assert not os.path.exists(os.path.join("plots"))


def test_experiment_wrapper_skip():
    model_lst = [
        "pytest_mnist_inn",
        "pytest_cifar_inn",
        "pytest_celeba_inn",
        "pytest_mnist_classic",
        "pytest_mnist_classic1024",
        "pytest_mnist_classicDeep1024",
        "pytest_mnist_classic2048",
        "pytest_cifar_classic",
        "pytest_celeba_classic",
    ]
    out = experiment_wrapper(model_lst, "pytest", skip_exec=True)
    assert isinstance(out, int)
    assert out == 0
