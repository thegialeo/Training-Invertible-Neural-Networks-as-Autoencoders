import pytest
import torch
from torchvision import datasets

from src.dataloader import get_loader, load_celeba, load_cifar, load_mnist


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


@pytest.mark.parametrize("load_dataset", [load_mnist, load_cifar, load_celeba])
def test_get_loader(load_dataset):
    trainset, testset = load_dataset()
    train_data, train_label = trainset[0]
    test_data, test_label = testset[0]
    trainloader = get_loader(trainset, 32, True)
    testloader = get_loader(testset, 32, False)
    train_data_batch, train_label_batch = next(iter(trainloader))
    test_data_batch, test_label_batch = next(iter(testloader))
    assert isinstance(trainloader, torch.utils.data.DataLoader)
    assert isinstance(testloader, torch.utils.data.DataLoader)
    assert len(trainloader) == int(len(trainset) / 32)
    assert len(testloader) == int(len(testset) / 32)
    assert isinstance(train_data_batch, torch.Tensor)
    assert isinstance(train_data_batch.dtype, type(torch.float32))
    assert isinstance(train_label_batch, torch.Tensor)
    assert isinstance(train_label_batch.dtype, type(torch.int64))
    assert train_data_batch.size() == (32,) + train_data.size()
    if isinstance(train_label, torch.Tensor):
        assert train_label_batch.size() == (32,) + train_label.size()
    elif isinstance(train_label, int):
        assert train_label_batch.size() == (32,)
    else:
        assert False
    assert isinstance(test_data_batch, torch.Tensor)
    assert isinstance(test_data_batch.dtype, type(torch.float32))
    assert isinstance(test_label_batch, torch.Tensor)
    assert isinstance(test_label_batch.dtype, type(torch.int64))
    assert test_data_batch.size() == (32,) + test_data.size()
    if isinstance(test_label, torch.Tensor):
        assert test_label_batch.size() == (32,) + test_label.size()
    elif isinstance(test_label, int):
        assert test_label_batch.size() == (32,)
    else:
        assert False
