"""Plotting and visualization functionalities."""
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.filemanager import create_folder


def plot_image(img: torch.Tensor, filename: str, folder="") -> None:
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
        os.path.join(path, filename) + ".png",
        transparent=True,
        bbox_inches="tight",
        pad_inches=0,
    )
