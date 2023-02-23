"""Loaders for datasets and training batches.

Functions:
    load_mnist() -> tuple[datasets.MNIST, datasets.MNIST]: load MNIST dataset
"""
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
