import torch

from src.functionalities import get_device


def test_get_device():
    if torch.cuda.is_available():
        # if gpu available
        gpu_device = get_device()
        assert isinstance(gpu_device, str)
        assert gpu_device == "cuda"
    else:
        # only cpu available
        cpu_device = get_device()
        assert isinstance(cpu_device, str)
        assert cpu_device == "cpu"
