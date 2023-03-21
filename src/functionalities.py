"""General funtionalities for training and evaluation.

Functions:
    get_device(int) -> str: get torch device name
    count_param(nn.Module) -> int: count the number of trainable parameters in a model
    init_weigths(nn.Module) -> None: initialize model weights
    get_model(int, str, dict) -> nn.Module: initialize and returns model
    get_optimizer(nn.Module, dict) -> torch.optimizer: initialize and returns optimizer
    plot_image(torch.Tensor, str, str) -> None: plot the torch tensor image
    plot_curves(list, list[Tensor], str, dict, str) -> None: plot curves
"""

import os
from typing import Union, cast

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.architecture import CLASSIC_ARCHITECTURES, INN_ARCHITECTURES
from src.filemanager import create_folder


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


def get_model(lat_dim: int, modelname: str, hyp_dict: dict) -> torch.nn.Module:
    """Get a model by name.

    Args:
        lat_dim (int): latent dimension
        modelname (str): model name
        hyp_dict (dict): collection of hyperparameters

    Returns:
        model (torch.nn.Module): model
    """
    if hyp_dict["INN"]:
        model = cast(torch.nn.Module, INN_ARCHITECTURES[modelname]())
        for key, param in model.named_parameters():
            split = key.split(".")
            if param.requires_grad:
                param.data = 0.1 * torch.randn(param.data.shape).cuda()
                if split[3][-1] == "3":
                    param.data.fill_(0.0)
    else:
        model = cast(torch.nn.Module, CLASSIC_ARCHITECTURES[modelname](lat_dim))
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


def plot_image(img: torch.Tensor, filename: str, folder: str = "") -> None:
    """Plot and save torch tensors representing images.

    Args:
        img (torch.Tensor): Image to plot
        filename (str): Filename to save the image
        folder (str): The folder to save the file in. Defaults to "".
    """
    img = torch.clamp(img, 0, 1)
    npimg = img.cpu().detach().numpy()
    plt.figsize = (30, 30)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

    subdir = "plots"
    path = os.path.join(subdir, folder) if folder else subdir

    create_folder(subdir)
    create_folder(path)

    plt.savefig(
        os.path.join(path, filename + ".png"),
        transparent=True,
        bbox_inches="tight",
        pad_inches=0,
    )


def plot_curves(
    xdata: list[list[Union[int, float]]],
    ydata: list[torch.Tensor],
    filename: str,
    plot_setting: dict,
    folder: str = "",
) -> None:
    """Plot curves.

    Args:
        xdata (list): x-axis data
        ydata (torch.Tensor): y-axis data
        filename (str): Filename to save the image
        plot_setting (dict): plot settings
        folder (str): The folder to save the file in. Defaults to "".
    """
    plt.rcParams.update({"font.size": 24})

    fig, axes = plt.subplots(1, 1, figsize=(15, 10))

    for idx, (x_values, y_values) in enumerate(zip(xdata, ydata)):
        axes.plot(x_values, y_values, label=plot_setting["names"][idx])

    axes.set_xlabel(plot_setting["x_label"])
    axes.set_ylabel(plot_setting["y_label"])
    axes.set_title(plot_setting["title"])
    axes.grid(True)
    axes.legend()

    plt.tight_layout()

    subdir = "plots"
    path = os.path.join(subdir, folder) if folder else subdir

    create_folder(subdir)
    create_folder(path)

    fig.savefig(
        os.path.join(path, filename + ".png"),
        transparent=True,
        bbox_inches="tight",
        pad_inches=0,
    )
