"""General funtionalities for training and evaluation.

Functions:
    get_device(int) -> str: get torch device name
    count_param(nn.Module) -> int: count the number of trainable parameters in a model
"""

from typing import cast

import torch

from src.architecture import CLASSIC_ARCHITECTURES, INN_ARCHITECTURES


def get_device(dev_idx: int = 0) -> str:
    """Use GPU if available, else CPU.

    Args:
        dev_idx (int): Which GPU to use. Defaults to 0.

    Returns:
        device (str): torch device name
    """
    if torch.cuda.is_available():
        torch.cuda.set_device(dev_idx)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    return device


def count_param(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in a model.

    Args:
        model (torch.nn.Module): model to count parameters

    Returns:
        num_params (int): number of trainable parameters
    """
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return num_params


def get_model(modelname: str, is_inn: bool, lat_dim: int) -> torch.nn.Module:
    """Get a model by name.

    Args:
        modelname (str): model name
        is_inn (bool): whether the model is an INN or not
        lat_dim (int): latent dimension

    Returns:
        model (torch.nn.Module): model
    """
    if is_inn:
        model = cast(torch.nn.Module, INN_ARCHITECTURES[modelname])
        for key, param in model.named_parameters():
            split = key.split(".")
            if param.requires_grad:
                param.data = 0.1 * torch.randn(param.data.shape).cuda()
                if split[3][-1] == "3":
                    param.data.fill_(0.0)
    else:
        model = cast(torch.nn.Module, CLASSIC_ARCHITECTURES[modelname](lat_dim))
        model(lat_dim)
        model.apply(init_weights)

    return model


def init_weights(model: torch.nn.Module) -> None:
    """Initialize the weights of a model.

    Args:
        model (torch.nn.Module): model to initialize weights
    """
    if isinstance(model, torch.nn.Linear):
        torch.nn.init.xavier_uniform(model.weight)
        model.bias.data.fill_(0.01)
