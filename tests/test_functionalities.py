import torch
import torchvision.models as models

from src.functionalities import count_param, get_device


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


def test_count_param():
    # https://pytorch.org/vision/main/models/generated/torchvision.models.alexnet.html
    # Number of parameters: 61100840
    alexnet = models.alexnet()
    num_params = count_param(alexnet)
    assert isinstance(num_params, int)
    assert num_params == 61100840
