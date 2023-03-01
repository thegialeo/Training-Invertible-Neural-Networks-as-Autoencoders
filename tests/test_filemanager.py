import os

import numpy as np

from src.filemanager import create_folder, delete_file, load_numpy, save_numpy


def test_create_folder():
    assert not os.path.exists("pytest")
    create_folder("pytest")
    assert os.path.exists("pytest")
    os.rmdir("pytest")
    assert not os.path.exists("pytest")


def test_numpy_IO():
    array = np.random.rand(5)
    save_numpy(array, "test")
    read_array = load_numpy("test")
    assert os.path.exists(os.path.join("log", "test.npy"))
    assert isinstance(array, np.ndarray)
    assert isinstance(read_array, np.ndarray)
    assert np.array_equal(read_array, array)
    delete_file("log", "test.npy")
    assert not os.path.exists(os.path.join("log", "test.npy"))
    if not os.listdir("log"):
        os.rmdir("log")


def test_numpy_IO_folder():
    array = np.random.rand(5)
    save_numpy(array, "test", "pytest")
    read_array = load_numpy("test", "pytest")
    assert os.path.exists(os.path.join("log", "pytest", "test.npy"))
    assert isinstance(array, np.ndarray)
    assert isinstance(read_array, np.ndarray)
    assert np.array_equal(read_array, array)
    delete_file("log", "test.npy", "pytest")
    assert not os.path.exists(os.path.join("log", "pytest", "test.npy"))
    assert os.path.exists(os.path.join("log", "pytest"))
    os.rmdir(os.path.join("log", "pytest"))
    assert not os.path.exists(os.path.join("log", "pytest"))
    if not os.listdir("log"):
        os.rmdir("log")
