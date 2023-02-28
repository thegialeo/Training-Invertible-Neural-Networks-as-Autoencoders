import pytest
import torch

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
from src.dataloader import get_loader, load_celeba, load_cifar, load_mnist


@pytest.mark.parametrize(
    "autoencoder",
    [
        MNISTAutoencoder,
        MNISTAutoencoder1024,
        MNISTAutoencoderDeep1024,
        MNISTAutoencoder2048,
    ],
)
def test_MNISTAutoencoder(autoencoder):
    trainset, testset = load_mnist()
    trainloader = get_loader(trainset, 2, True)
    testloader = get_loader(testset, 2, False)
    train_data_batch, _ = next(iter(trainloader))
    test_data_batch, _ = next(iter(testloader))
    model = autoencoder(12)
    out_train = model(train_data_batch)
    out_test = model(test_data_batch)
    assert isinstance(train_data_batch, torch.Tensor)
    assert isinstance(train_data_batch.dtype, type(torch.float32))
    assert isinstance(out_train, torch.Tensor)
    assert isinstance(out_train.dtype, type(torch.float32))
    assert isinstance(test_data_batch, torch.Tensor)
    assert isinstance(test_data_batch.dtype, type(torch.float32))
    assert isinstance(out_test, torch.Tensor)
    assert isinstance(out_test.dtype, type(torch.float32))
    assert train_data_batch.size() == (2, 1, 28, 28)
    assert out_train.size() == (2, 1, 28, 28)
    assert test_data_batch.size() == (2, 1, 28, 28)
    assert out_test.size() == (2, 1, 28, 28)


def test_MNIST_INN_Autoencoder():
    trainset, testset = load_mnist()
    trainloader = get_loader(trainset, 2, True)
    testloader = get_loader(testset, 2, False)
    train_data_batch, _ = next(iter(trainloader))
    test_data_batch, _ = next(iter(testloader))
    model = get_mnist_inn_autoencoder()
    lat_train = model(train_data_batch)
    lat_test = model(test_data_batch)
    out_train = model(lat_train, rev=True)
    out_test = model(lat_test, rev=True)
    assert isinstance(train_data_batch, torch.Tensor)
    assert isinstance(train_data_batch.dtype, type(torch.float32))
    assert isinstance(lat_train, torch.Tensor)
    assert isinstance(lat_train.dtype, type(torch.float32))
    assert isinstance(out_train, torch.Tensor)
    assert isinstance(out_train.dtype, type(torch.float32))
    assert isinstance(test_data_batch, torch.Tensor)
    assert isinstance(test_data_batch.dtype, type(torch.float32))
    assert isinstance(lat_test, torch.Tensor)
    assert isinstance(lat_test.dtype, type(torch.float32))
    assert isinstance(out_test, torch.Tensor)
    assert isinstance(out_test.dtype, type(torch.float32))
    assert train_data_batch.size() == (2, 1, 28, 28)
    assert lat_train.size() == (2, 1, 28, 28)
    assert out_train.size() == (2, 1, 28, 28)
    assert test_data_batch.size() == (2, 1, 28, 28)
    assert lat_test.size() == (2, 1, 28, 28)
    assert out_test.size() == (2, 1, 28, 28)


def test_CIFAR10Autoencoder():
    trainset, testset = load_cifar()
    trainloader = get_loader(trainset, 2, True)
    testloader = get_loader(testset, 2, False)
    train_data_batch, _ = next(iter(trainloader))
    test_data_batch, _ = next(iter(testloader))
    model = CIFAR10Autoencoder(250)
    out_train = model(train_data_batch)
    out_test = model(test_data_batch)
    assert isinstance(train_data_batch, torch.Tensor)
    assert isinstance(train_data_batch.dtype, type(torch.float32))
    assert isinstance(out_train, torch.Tensor)
    assert isinstance(out_train.dtype, type(torch.float32))
    assert isinstance(test_data_batch, torch.Tensor)
    assert isinstance(test_data_batch.dtype, type(torch.float32))
    assert isinstance(out_test, torch.Tensor)
    assert isinstance(out_test.dtype, type(torch.float32))
    assert train_data_batch.size() == (2, 3, 32, 32)
    assert out_train.size() == (2, 3, 32, 32)
    assert test_data_batch.size() == (2, 3, 32, 32)
    assert out_test.size() == (2, 3, 32, 32)


def test_CIFAR_INN_Autoencoder():
    trainset, testset = load_cifar()
    trainloader = get_loader(trainset, 2, True)
    testloader = get_loader(testset, 2, False)
    train_data_batch, _ = next(iter(trainloader))
    test_data_batch, _ = next(iter(testloader))
    model = get_cifar10_inn_autoencoder()
    lat_train = model(train_data_batch)
    lat_test = model(test_data_batch)
    out_train = model(lat_train, rev=True)
    out_test = model(lat_test, rev=True)
    assert isinstance(train_data_batch, torch.Tensor)
    assert isinstance(train_data_batch.dtype, type(torch.float32))
    assert isinstance(lat_train, torch.Tensor)
    assert isinstance(lat_train.dtype, type(torch.float32))
    assert isinstance(out_train, torch.Tensor)
    assert isinstance(out_train.dtype, type(torch.float32))
    assert isinstance(test_data_batch, torch.Tensor)
    assert isinstance(test_data_batch.dtype, type(torch.float32))
    assert isinstance(lat_test, torch.Tensor)
    assert isinstance(lat_test.dtype, type(torch.float32))
    assert isinstance(out_test, torch.Tensor)
    assert isinstance(out_test.dtype, type(torch.float32))
    assert train_data_batch.size() == (2, 3, 32, 32)
    assert lat_train.size() == (2, 3, 32, 32)
    assert out_train.size() == (2, 3, 32, 32)
    assert test_data_batch.size() == (2, 3, 32, 32)
    assert lat_test.size() == (2, 3, 32, 32)
    assert out_test.size() == (2, 3, 32, 32)


def test_CelebAAutoencoder():
    trainset, testset = load_celeba(resize=False)
    trainloader = get_loader(trainset, 2, True)
    testloader = get_loader(testset, 2, False)
    train_data_batch, _ = next(iter(trainloader))
    test_data_batch, _ = next(iter(testloader))
    model = CelebAAutoencoder(250)
    out_train = model(train_data_batch)
    out_test = model(test_data_batch)
    assert isinstance(train_data_batch, torch.Tensor)
    assert isinstance(train_data_batch.dtype, type(torch.float32))
    assert isinstance(out_train, torch.Tensor)
    assert isinstance(out_train.dtype, type(torch.float32))
    assert isinstance(test_data_batch, torch.Tensor)
    assert isinstance(test_data_batch.dtype, type(torch.float32))
    assert isinstance(out_test, torch.Tensor)
    assert isinstance(out_test.dtype, type(torch.float32))
    assert train_data_batch.size() == (2, 3, 218, 178)
    assert out_train.size() == (2, 3, 218, 178)
    assert test_data_batch.size() == (2, 3, 218, 178)
    assert out_test.size() == (2, 3, 218, 178)


def test_CelebA_INN_Autoencoder():
    trainset, testset = load_celeba(resize=True)
    trainloader = get_loader(trainset, 2, True)
    testloader = get_loader(testset, 2, False)
    train_data_batch, _ = next(iter(trainloader))
    test_data_batch, _ = next(iter(testloader))
    model = get_celeba_inn_autoencoder()
    lat_train = model(train_data_batch)
    lat_test = model(test_data_batch)
    out_train = model(lat_train, rev=True)
    out_test = model(lat_test, rev=True)
    assert isinstance(train_data_batch, torch.Tensor)
    assert isinstance(train_data_batch.dtype, type(torch.float32))
    assert isinstance(lat_train, torch.Tensor)
    assert isinstance(lat_train.dtype, type(torch.float32))
    assert isinstance(out_train, torch.Tensor)
    assert isinstance(out_train.dtype, type(torch.float32))
    assert isinstance(test_data_batch, torch.Tensor)
    assert isinstance(test_data_batch.dtype, type(torch.float32))
    assert isinstance(lat_test, torch.Tensor)
    assert isinstance(lat_test.dtype, type(torch.float32))
    assert isinstance(out_test, torch.Tensor)
    assert isinstance(out_test.dtype, type(torch.float32))
    assert train_data_batch.size() == (2, 3, 128, 128)
    assert lat_train.size() == (2, 3, 128, 128)
    assert out_train.size() == (2, 3, 128, 128)
    assert test_data_batch.size() == (2, 3, 128, 128)
    assert lat_test.size() == (2, 3, 128, 128)
    assert out_test.size() == (2, 3, 128, 128)
