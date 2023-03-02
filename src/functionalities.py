"""General funtionalities for training and evaluation.

Functions:
    get_device(int) -> str: get torch device name
    count_param(nn.Module) -> int: count the number of trainable parameters in a model
"""

import torch


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
