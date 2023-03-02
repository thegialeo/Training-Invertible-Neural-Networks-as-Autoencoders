"""Functions for saving/reading data and models.

Functions:
    create_folder(str) -> None: create a folder.
    delete_file(str, str, str) -> None: delete a file.
    save_numpy(ndarray, str, str) -> None: save numpy array to file.
    load_numpy(str, str) -> ndarray: load numpy array from file.
    save_model(model str, str) -> None: save model to file.
    load_model(model, str, str) -> model: load model from file.
"""

import os
from typing import Union

import numpy as np
import torch

from FrEIA.framework import ReversibleGraphNet


def create_folder(folder: str) -> None:
    """
    Create a folder.

    Args:
        folder (str): The folder to create.
    """
    if not os.path.exists(folder):
        os.makedirs(folder)


def delete_file(subdir: str, filename: str, folder: str = "") -> None:
    """
    Delete a file.

    Args:
        subdir (str): The folder to delete the file from.
        filename (str): The name of the file.
        folder (str): The folder to delete the file from. Defaults to "".
    """
    path = os.path.join(subdir, folder) if folder else subdir

    if os.path.isfile(os.path.join(path, filename)):
        os.remove(os.path.join(path, filename))


def save_numpy(array: np.ndarray, filename: str, folder: str = "") -> None:
    """
    Save numpy array to a file wrapper.

    Args:
        array (ndarray): Numpy array to save.
        filename (str): The name of the file.
        folder (str): The folder to save the file in. Defaults to "".
    """
    subdir = "logs"
    path = os.path.join(subdir, folder) if folder else subdir

    create_folder(subdir)
    create_folder(path)

    with open(os.path.join(path, filename + ".npy"), "wb") as file:
        np.save(file, array)


def load_numpy(filename: str, folder: str = "") -> np.ndarray:
    """
    Load numpy array from a file wrapper.

    Args:
        filename (str): The name of the file.
        folder (str): The folder to load the file from. Defaults to "".

    Returns:
        array (ndarray): Loaded numpy array.
    """
    subdir = "logs"
    path = os.path.join(subdir, folder) if folder else subdir

    with open(os.path.join(path, filename + ".npy"), "rb") as file:
        array = np.load(file)

    return array


def save_model(
    model: Union[torch.nn.Module, ReversibleGraphNet], filename: str, folder: str = ""
) -> None:
    """Save model in a file.

    Args:
        model (nn.Module | ReversibleGraphNet): The model to save.
        filename (str): The name of the file.
        folder (str): The folder to save the file in. Defaults to "".
    """
    subdir = "models"
    path = os.path.join(subdir, folder) if folder else subdir

    create_folder(subdir)
    create_folder(path)

    model.cpu()
    torch.save(model.state_dict(), os.path.join(path, filename + ".pt"))


def load_model(
    model: Union[torch.nn.Module, ReversibleGraphNet], filename: str, folder: str = ""
) -> Union[torch.nn.Module, ReversibleGraphNet]:
    """Load model from a file.

    Args:
        model (nn.Module | ReversibleGraphNet): Architecture of loaded model.
        filename (str): The name of the file.
        folder (str): The folder to load the file from. Defaults to "".

    Returns:
        model (nn.Module | ReversibleGraphNet): Loaded model.
    """
    subdir = "models"
    path = os.path.join(subdir, folder) if folder else subdir

    model.load_state_dict(
        torch.load(os.path.join(path, filename + ".pt"), map_location="cpu")
    )

    return model
