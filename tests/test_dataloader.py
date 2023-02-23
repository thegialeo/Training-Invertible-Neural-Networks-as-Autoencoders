import torch
from torchvision import datasets

from src.dataloader import load_celeba, load_cifar, load_mnist


def test_load_mnist():
    trainset, testset = load_mnist()
    train_data, train_label = trainset[0]
    test_data, test_label = testset[0]
    assert isinstance(trainset, datasets.MNIST)
    assert isinstance(testset, datasets.MNIST)
    assert len(trainset) == 60000
    assert len(testset) == 10000
    assert isinstance(train_data, torch.Tensor)
    assert isinstance(train_data.dtype, type(torch.float32))
    assert isinstance(train_label, int)
    assert train_data.size() == (1, 28, 28)
    assert isinstance(test_data, torch.Tensor)
    assert isinstance(test_data.dtype, type(torch.float32))
    assert isinstance(test_label, int)
    assert test_data.size() == (1, 28, 28)


def test_load_cifar():
    trainset, testset = load_cifar()
    train_data, train_label = trainset[0]
    test_data, test_label = testset[0]
    assert isinstance(trainset, datasets.CIFAR10)
    assert isinstance(testset, datasets.CIFAR10)
    assert len(trainset) == 50000
    assert len(testset) == 10000
    assert isinstance(train_data, torch.Tensor)
    assert isinstance(train_data.dtype, type(torch.float32))
    assert isinstance(train_label, int)
    assert train_data.size() == (3, 32, 32)
    assert isinstance(test_data, torch.Tensor)
    assert isinstance(test_data.dtype, type(torch.float32))
    assert isinstance(test_label, int)
    assert test_data.size() == (3, 32, 32)


def test_load_celeba():
    trainset, testset = load_celeba()
    train_data, train_label = trainset[0]
    test_data, test_label = testset[0]
    assert isinstance(trainset, datasets.CelebA)
    assert isinstance(testset, datasets.CelebA)
    assert len(trainset) == 162770
    assert len(testset) == 19962
    assert isinstance(train_data, torch.Tensor)
    assert isinstance(train_data.dtype, type(torch.float32))
    assert isinstance(train_label, torch.Tensor)
    assert isinstance(train_label.dtype, type(torch.int64))
    assert train_data.size() == (3, int(128 * 218 / 178), 128)
    assert train_label.size() == (40,)
    assert isinstance(test_data, torch.Tensor)
    assert isinstance(test_data.dtype, type(torch.float32))
    assert isinstance(test_label, torch.Tensor)
    assert isinstance(test_label.dtype, type(torch.int64))
    assert test_data.size() == (3, int(128 * 218 / 178), 128)
    assert test_label.size() == (40,)
