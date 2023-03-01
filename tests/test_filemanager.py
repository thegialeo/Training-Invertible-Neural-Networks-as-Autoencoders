import os

from src.filemanager import create_folder


def test_create_folder():
    assert not os.path.exists("./pytest")
    create_folder("./pytest")
    assert os.path.exists("./pytest")
    os.rmdir("./pytest")
    assert not os.path.exists("./pytest")
