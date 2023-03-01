"""Functions for saving/reading data and models.

Functions:
    save_numpy(ndarray, str, str) -> None: save numpy array to file.
    load_numpy(str, str) -> ndarray: load numpy array from file.
    create_folder(str) -> None: create a folder.
    delete_file(str, str, str) -> None: delete a file.
"""

import os

import numpy as np


def save_numpy(array: np.ndarray, filename: str, folder: str = "") -> None:
    """
    Save numpy array to a file.

    Args:
        array (ndarray): Numpy array to save.
        filename (str): The name of the file.
        folder (str): The folder to save the file in. Defaults to "".
    """
    subdir = "./variables"
    path = os.path.join(subdir, folder) if folder else subdir

    create_folder(subdir)
    create_folder(path)

    with open(os.path.join(path, filename + ".npy"), "wb") as file:
        np.save(file, array)


def load_numpy(filename: str, folder: str = "") -> np.ndarray:
    """
    Load numpy array from a file.

    Args:
        filename (str): The name of the file.
        folder (str): The folder to load the file from. Defaults to "".

    Returns:
        array (ndarray): Numpy array.
    """
    subdir = "./variables"
    path = os.path.join(subdir, folder) if folder else subdir

    with open(os.path.join(path, filename + ".npy"), "rb") as file:
        array = np.load(file)

    return array


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
