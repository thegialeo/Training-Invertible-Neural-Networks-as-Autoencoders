"""Hyperparameters settings."""

import torch

# ---------------------------------------------------------------------------- #
#                                INN Autoencoder                               #
# ---------------------------------------------------------------------------- #

MNIST_INN_HYPERPARAMETER = {
    "num_epoch": 10,
    "batch_size": 128,
    "lat_dim_lst": [1, 2, 4, 8, 16, 32, 64],
    "lr": 1e-3,
    "betas": (0.8, 0.8),
    "eps": 1e-4,
    "weight_decay": 1e-6,
    "milestone": [8, 10],
    "a_rec": 1,
    "a_dist": 0,
    "a_sparse": 1,
    "dtype": torch.float32,
    "INN": True,
}

CIFAR_INN_HYPERPARAMETER = {
    "num_epoch": 15,
    "batch_size": 128,
    "lat_dim_lst": [2**x for x in range(11)],
    "lr": 1e-3,
    "betas": (0.8, 0.8),
    "eps": 1e-4,
    "weight_decay": 1e-6,
    "milestone": [8, 10],
    "a_rec": 1,
    "a_dist": 0,
    "a_sparse": 1,
    "dtype": torch.float32,
    "INN": True,
}

CELEBA_INN_HYPERPARAMETER = {
    "num_epoch": 8,
    "batch_size": 32,
    "lat_dim_lst": [2**x for x in range(0, 11, 2)],
    "lr": 1e-3,
    "betas": (0.8, 0.8),
    "eps": 1e-4,
    "weight_decay": 1e-6,
    "milestone": [6, 7, 8],
    "modelname": "celeba_inn",
    "a_rec": 1,
    "a_dist": 0,
    "a_sparse": 1,
    "dtype": torch.float32,
    "INN": True,
}


# ---------------------------------------------------------------------------- #
#                              Classic Autoencoder                             #
# ---------------------------------------------------------------------------- #

MNIST_CLASSIC_HYPERPARAMETER = {
    "num_epoch": 100,
    "batch_size": 128,
    "lat_dim_lst": [1, 2, 4, 8, 16, 32, 64],
    "lr": 1e-3,
    "betas": (0.9, 0.999),
    "eps": 1e-8,
    "weight_decay": 1e-5,
    "milestone": [10 * x for x in range(1, 11)],
    "dtype": torch.float32,
    "INN": False,
}

CIFAR_CLASSIC_HYPERPARAMETER = {
    "num_epoch": 100,
    "batch_size": 128,
    "lat_dim_lst": [2**x for x in range(11)],
    "lr": 1e-3,
    "betas": (0.9, 0.999),
    "eps": 1e-8,
    "weight_decay": 1e-5,
    "milestone": [60, 85, 100],
    "dtype": torch.float32,
    "INN": False,
}

CELEBA_CLASSIC_HYPERPARAMETER = {
    "num_epoch": 10,
    "batch_size": 32,
    "lat_dim_lst": [2**x for x in range(0, 11, 2)],
    "lr": 1e-3,
    "betas": (0.9, 0.999),
    "eps": 1e-8,
    "weight_decay": 1e-5,
    "milestone": [8, 9, 10],
    "dtype": torch.float32,
    "INN": False,
}

# ---------------------------------------------------------------------------- #
#                                  Entry point                                 #
# ---------------------------------------------------------------------------- #

HYPERPARAMETER = {
    "mnist_inn": MNIST_INN_HYPERPARAMETER,
    "cifar_inn": CIFAR_INN_HYPERPARAMETER,
    "celeba_inn": CELEBA_INN_HYPERPARAMETER,
    "mnist_classic": MNIST_CLASSIC_HYPERPARAMETER,
    "mnist_classic1024": MNIST_CLASSIC_HYPERPARAMETER,
    "mnist_classicDeep1024": MNIST_CLASSIC_HYPERPARAMETER,
    "mnist_classic2048": MNIST_CLASSIC_HYPERPARAMETER,
    "cifar_classic": CIFAR_CLASSIC_HYPERPARAMETER,
    "celeba_classic": CELEBA_CLASSIC_HYPERPARAMETER,
}
