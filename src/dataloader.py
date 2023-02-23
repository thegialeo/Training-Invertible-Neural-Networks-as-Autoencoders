"""Loaders for datasets and training batches.

Functions:
    load_mnist() -> tuple[datasets.MNIST, datasets.MNIST]: load MNIST dataset
    load_cifar() -> tuple[datasets.CIFAR10, datasets.CIFAR10]: load CIFAR10 dataset
    load_celeba() -> tuple[datasets.CelebA, datasets.CelebA]: load CelebA dataset
    get_loader(datasets, int, bool) -> torch.DataLoader: get torch dataloader from dataset
"""
import torch
from torchvision import datasets, transforms


def load_mnist() -> tuple[datasets.MNIST, datasets.MNIST]:
    """Load MNIST dataset.

    Returns:
        trainset (datasets.MNIST): training dataset
        testset (datasets.MNIST): test dataset
    """
    save_path = "./datasets/mnist"

    transform = transforms.Compose([transforms.ToTensor()])

    trainset = datasets.MNIST(save_path, train=True, download=True, transform=transform)
    testset = datasets.MNIST(save_path, train=False, download=True, transform=transform)

    return trainset, testset


def load_cifar() -> tuple[datasets.CIFAR10, datasets.CIFAR10]:
    """Load CIFAR10 dataset.

    Returns:
        trainset (datasets.CIFAR10): training dataset
        testset (datasets.CIFAR10): test dataset
    """
    save_path = "./datasets/cifar10"

    transform = transforms.Compose([transforms.ToTensor()])

    trainset = datasets.CIFAR10(
        save_path, train=True, download=True, transform=transform
    )
    testset = datasets.CIFAR10(
        save_path, train=False, download=True, transform=transform
    )

    return trainset, testset


def load_celeba() -> tuple[datasets.CelebA, datasets.CelebA]:
    """Load CelebA dataset.

    Returns:
        trainset (datasets.CelebA): training dataset
        testset (datasets.CelebA): test dataset
    """
    save_path = "./datasets/celeba"

    transform = transforms.Compose([transforms.Resize(128), transforms.ToTensor()])

    trainset = datasets.CelebA(
        save_path, split="train", download=True, transform=transform
    )
    testset = datasets.CelebA(
        save_path, split="test", download=True, transform=transform
    )

    return trainset, testset


def get_loader(
    dataset: datasets, batch_size: int, shuffle: bool
) -> torch.utils.data.DataLoader:
    """Get torch dataloader from dataset.

    Args:
        dataset (datasets): dataset
        batch_size (int): batch size
        shuffle (bool): shuffle dataset each epoch

    Returns:
        loader (torch.utils.data.DataLoader): data loader
    """
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True, drop_last=True
    )
    return loader
