import os

import pytest
import torch

from src.dataloader import load_celeba, load_cifar, load_mnist
from src.filemanager import delete_file
from src.plot import plot_image


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
