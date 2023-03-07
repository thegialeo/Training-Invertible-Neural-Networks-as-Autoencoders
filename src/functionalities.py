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


def init_weights(model: torch.nn.Module) -> None:
    """Initialize the weights of a model.

    Args:
        model (torch.nn.Module): model to initialize weights
    """
    if isinstance(model, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(model.weight)
        model.bias.data.fill_(0.01)


def get_model(lat_dim: int, hyp_dict: dict) -> torch.nn.Module:
    """Get a model by name.

    Args:
        lat_dim (int): latent dimension
        hyp_dict (dict): collection of hyperparameters

    Returns:
        model (torch.nn.Module): model
    """
    if hyp_dict["INN"]:
        model = cast(torch.nn.Module, INN_ARCHITECTURES[hyp_dict["modelname"]]())
        for key, param in model.named_parameters():
            split = key.split(".")
            if param.requires_grad:
                param.data = 0.1 * torch.randn(param.data.shape).cuda()
                if split[3][-1] == "3":
                    param.data.fill_(0.0)
    else:
        model = cast(
            torch.nn.Module, CLASSIC_ARCHITECTURES[hyp_dict["modelname"]](lat_dim)
        )
        model.apply(init_weights)

    return model


def get_optimizer(model: torch.nn.Module, hyp_dict: dict) -> torch.optim.Optimizer:
    """Get optimizer and learning rate scheduler.

    Args:
        model (torch.nn.Module): model
        hyp_dict (dict): collection of hyperparameters

    Returns:
        optimizer (torch.optim.Optimizer): optimizer
    """
    model_params = []
    for param in model.parameters():
        if param.requires_grad:
            model_params.append(param)

    optimizer = cast(
        torch.optim.Optimizer,
        torch.optim.Adam(
            model_params,
            lr=hyp_dict["lr"],
            betas=hyp_dict["betas"],
            eps=hyp_dict["eps"],
            weight_decay=hyp_dict["weight_decay"],
        ),
    )

    return optimizer
