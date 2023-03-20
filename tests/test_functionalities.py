import math
import os

import numpy as np
import pytest
import torch
import torchvision.models as models

from src.architecture import CLASSIC_ARCHITECTURES
from src.dataloader import load_celeba, load_cifar, load_mnist
from src.filemanager import delete_file
from src.functionalities import (
    count_param,
    get_device,
    get_model,
    get_optimizer,
    init_weights,
    plot_curves,
    plot_image,
)
from src.settings import HYPERPARAMETER


def test_get_device():
    if torch.cuda.is_available():
        # if gpu available
        gpu_device = get_device()
        assert isinstance(gpu_device, str)
        assert gpu_device == "cuda"
    else:
        # only cpu available
        cpu_device = get_device()
        assert isinstance(cpu_device, str)
        assert cpu_device == "cpu"


def test_count_param():
    # https://pytorch.org/vision/main/models/generated/torchvision.models.alexnet.html
    # Number of parameters: 61100840
    alexnet = models.alexnet()
    num_params = count_param(alexnet)
    assert isinstance(num_params, int)
    assert num_params == 61100840


def test_init_weights():
    model = CLASSIC_ARCHITECTURES["mnist_classic"](5)
    model_params = model.parameters()
    model.apply(init_weights)
    new_model_params = model.parameters()
    assert model_params != new_model_params


@pytest.mark.parametrize("modelname", ["mnist_inn", "cifar_inn", "celeba_inn"])
def test_get_model_optimizer_inn(modelname):
    hyp_dict = HYPERPARAMETER[modelname]
    model = get_model(5, modelname, hyp_dict)
    optimizer = get_optimizer(model, hyp_dict)
    assert isinstance(model, torch.nn.Module)
    assert isinstance(optimizer, torch.optim.Adam)


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
def test_get_model_optimizer_classic(modelname):
    hyp_dict = HYPERPARAMETER[modelname]
    model = get_model(5, modelname, hyp_dict)
    optimizer = get_optimizer(model, hyp_dict)
    assert isinstance(model, torch.nn.Module)
    assert isinstance(optimizer, torch.optim.Adam)


@pytest.mark.parametrize("load_data", [load_mnist, load_cifar])
def test_plot_image_mnist_cifar(load_data):
    trainset, _ = load_data()
    img, _ = trainset[0]
    plot_image(img, "test")
    assert isinstance(img, torch.Tensor)
    assert os.path.exists(os.path.join("plots", "test.png"))
    delete_file("plots", "test.png")
    assert not os.path.exists(os.path.join("plots", "test.png"))
    assert os.path.exists("plots")
    if not os.listdir("plots"):
        os.rmdir("plots")
        assert not os.path.exists("plots")


@pytest.mark.parametrize("load_data", [load_mnist, load_cifar])
def test_plot_image_mnist_cifar_folder(load_data):
    trainset, _ = load_data()
    img, _ = trainset[0]
    plot_image(img, "test", "pytest")
    assert isinstance(img, torch.Tensor)
    assert os.path.exists(os.path.join("plots", "pytest", "test.png"))
    delete_file("plots", "test.png", "pytest")
    assert not os.path.exists(os.path.join("plots", "pytest", "test.png"))
    assert os.path.exists(os.path.join("plots", "pytest"))
    os.rmdir(os.path.join("plots", "pytest"))
    assert not os.path.exists(os.path.join("plots", "pytest"))
    assert os.path.exists("plots")
    if not os.listdir("plots"):
        os.rmdir("plots")
        assert not os.path.exists("plots")


@pytest.mark.parametrize("resize", [True, False])
def test_plot_image_celeba(resize):
    trainset, _ = load_celeba(resize)
    img, _ = trainset[0]
    plot_image(img, "test")
    assert isinstance(img, torch.Tensor)
    assert os.path.exists(os.path.join("plots", "test.png"))
    delete_file("plots", "test.png")
    assert not os.path.exists(os.path.join("plots", "test.png"))
    assert os.path.exists("plots")
    if not os.listdir("plots"):
        os.rmdir("plots")
        assert not os.path.exists("plots")


@pytest.mark.parametrize("resize", [True, False])
def test_plot_image_celeba_folder(resize):
    trainset, _ = load_celeba(resize)
    img, _ = trainset[0]
    plot_image(img, "test", "pytest")
    assert isinstance(img, torch.Tensor)
    assert os.path.exists(os.path.join("plots", "pytest", "test.png"))
    delete_file("plots", "test.png", "pytest")
    assert not os.path.exists(os.path.join("plots", "pytest", "test.png"))
    assert os.path.exists(os.path.join("plots", "pytest"))
    os.rmdir(os.path.join("plots", "pytest"))
    assert not os.path.exists(os.path.join("plots", "pytest"))
    assert os.path.exists("plots")
    if not os.listdir("plots"):
        os.rmdir("plots")
        assert not os.path.exists("plots")


def test_plot_curves():
    x = [x for x in np.arange(0, 12, 0.1)]
    y_sin = torch.zeros(len(x))
    y_cos = torch.zeros(len(x))

    for i, x_i in enumerate(x):
        y_sin[i] = math.sin(x_i)
        y_cos[i] = math.cos(x_i)

    plot_settings = {
        "names": ["sine", "cosine"],
        "x_label": "angle",
        "y_label": "function",
        "title": "Sine and Cosine Functions",
    }

    plot_curves(x, [y_sin, y_cos], "test", plot_settings, "pytest")

    assert os.path.exists(os.path.join("plots", "pytest", "test.png"))
    delete_file("plots", "test.png", "pytest")
    assert not os.path.exists(os.path.join("plots", "pytest", "test.png"))
    assert os.path.exists(os.path.join("plots", "pytest"))
    os.rmdir(os.path.join("plots", "pytest"))
    assert not os.path.exists(os.path.join("plots", "pytest"))
    assert os.path.exists("plots")
    if not os.listdir("plots"):
        os.rmdir("plots")
        assert not os.path.exists("plots")
