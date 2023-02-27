import pytest
import torch

from src.architecture import (
    MNISTAutoencoder,
    MNISTAutoencoder1024,
    MNISTAutoencoder2048,
    MNISTAutoencoderDeep1024,
)
from src.dataloader import get_loader, load_mnist


@pytest.mark.parametrize("autoencoder", [MNISTAutoencoder, MNISTAutoencoder1024, MNISTAutoencoderDeep1024, MNISTAutoencoder2048])
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
