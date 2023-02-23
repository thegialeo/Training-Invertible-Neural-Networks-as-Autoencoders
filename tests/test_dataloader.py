import torch
from torchvision import datasets

from src.dataloader import load_mnist


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
