import os

import numpy as np
import pytest

from src.architecture import (
    CelebAAutoencoder,
    CIFAR10Autoencoder,
    MNISTAutoencoder,
    MNISTAutoencoder1024,
    MNISTAutoencoder2048,
    MNISTAutoencoderDeep1024,
    get_celeba_inn_autoencoder,
    get_cifar10_inn_autoencoder,
    get_mnist_inn_autoencoder,
)
from src.filemanager import (
    create_folder,
    delete_file,
    load_model,
    load_numpy,
    save_model,
    save_numpy,
)


def test_create_folder():
    assert not os.path.exists("pytest")
    create_folder("pytest")
    assert os.path.exists("pytest")
    os.rmdir("pytest")
    assert not os.path.exists("pytest")


def test_numpy_IO():
    array = np.random.rand(5)
    save_numpy(array, "test")
    read_array = load_numpy("test")
    assert isinstance(array, np.ndarray)
    assert isinstance(read_array, np.ndarray)
    assert os.path.exists(os.path.join("logs", "test.npy"))
    assert np.array_equal(read_array, array)
    delete_file("logs", "test.npy")
    assert not os.path.exists(os.path.join("logs", "test.npy"))
    assert os.path.exists("logs")
    if not os.listdir("logs"):
        os.rmdir("logs")
        assert not os.path.exists("logs")


def test_numpy_IO_folder():
    array = np.random.rand(5)
    save_numpy(array, "test", "pytest")
    read_array = load_numpy("test", "pytest")
    assert isinstance(array, np.ndarray)
    assert isinstance(read_array, np.ndarray)
    assert os.path.exists(os.path.join("logs", "pytest", "test.npy"))
    assert np.array_equal(read_array, array)
    delete_file("logs", "test.npy", "pytest")
    assert not os.path.exists(os.path.join("logs", "pytest", "test.npy"))
    assert os.path.exists(os.path.join("logs", "pytest"))
    os.rmdir(os.path.join("logs", "pytest"))
    assert not os.path.exists(os.path.join("logs", "pytest"))
    assert os.path.exists("logs")
    if not os.listdir("logs"):
        os.rmdir("logs")
        assert not os.path.exists("logs")


@pytest.mark.parametrize(
    "architecture",
    [
        MNISTAutoencoder(12),
        MNISTAutoencoder1024(12),
        MNISTAutoencoderDeep1024(12),
        MNISTAutoencoder2048(12),
        CIFAR10Autoencoder(250),
        CelebAAutoencoder(250),
        get_mnist_inn_autoencoder(),
        get_cifar10_inn_autoencoder(),
        get_celeba_inn_autoencoder(),
    ],
)
def test_model_IO(architecture):
    model = architecture
    save_model(model, "test")
    read_model = load_model(model, "test")
    assert isinstance(read_model, type(model))
    assert os.path.exists(os.path.join("models", "test.pt"))
    delete_file("models", "test.pt")
    assert not os.path.exists(os.path.join("models", "test.pt"))
    assert os.path.exists("models")
    if not os.listdir("models"):
        os.rmdir("models")
        assert not os.path.exists("models")


@pytest.mark.parametrize(
    "architecture",
    [
        MNISTAutoencoder(12),
        MNISTAutoencoder1024(12),
        MNISTAutoencoderDeep1024(12),
        MNISTAutoencoder2048(12),
        CIFAR10Autoencoder(250),
        CelebAAutoencoder(250),
        get_mnist_inn_autoencoder(),
        get_cifar10_inn_autoencoder(),
        get_celeba_inn_autoencoder(),
    ],
)
def test_model_IO_folder(architecture):
    model = architecture
    save_model(model, "test", "pytest")
    read_model = load_model(model, "test", "pytest")
    assert isinstance(read_model, type(model))
    assert os.path.exists(os.path.join("models", "pytest", "test.pt"))
    delete_file("models", "test.pt", "pytest")
    assert not os.path.exists(os.path.join("models", "pytest", "test.pt"))
    assert os.path.exists(os.path.join("models", "pytest"))
    os.rmdir(os.path.join("models", "pytest"))
    assert not os.path.exists(os.path.join("models", "pytest"))
    assert os.path.exists("models")
    if not os.listdir("models"):
        os.rmdir("models")
        assert not os.path.exists("models")
