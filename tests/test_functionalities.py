import pytest
import torch
import torchvision.models as models

from src.architecture import CLASSIC_ARCHITECTURES
from src.functionalities import (
    count_param,
    get_device,
    get_model,
    get_optimizer,
    init_weights,
)
from src.settings import HYPERPARAMETER


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


def test_init_weights():
    model = CLASSIC_ARCHITECTURES["mnist_classic"](5)
    model_params = model.parameters()
    model.apply(init_weights)
    new_model_params = model.parameters()
    assert model_params != new_model_params


@pytest.mark.parametrize("modelname", ["mnist_inn", "cifar_inn", "celeba_inn"])
def test_get_model_optimizer_inn(modelname):
    hyp_dict = HYPERPARAMETER[modelname]
    model = get_model(5, modelname, hyp_dict)
    optimizer = get_optimizer(model, hyp_dict)
    assert isinstance(model, torch.nn.Module)
    assert isinstance(optimizer, torch.optim.Adam)


@pytest.mark.parametrize(
    "modelname",
    [
        "mnist_classic",
        "mnist_classic1024",
        "mnist_classicDeep1024",
        "mnist_classic2048",
        "cifar_classic",
        "celeba_classic",
    ],
)
def test_get_model_optimizer_classic(modelname):
    hyp_dict = HYPERPARAMETER[modelname]
    model = get_model(5, modelname, hyp_dict)
    optimizer = get_optimizer(model, hyp_dict)
    assert isinstance(model, torch.nn.Module)
    assert isinstance(optimizer, torch.optim.Adam)
