import torch


def get_device(dev_idx: int = 0) -> str:
    """Use GPU if available, else CPU

    Args:
        dev_idx (int, optional): Which GPU to use. Defaults to 0.

    Returns:
        device (str): torch device name
    """

    if torch.cuda.is_available():
        torch.cuda.set_device(dev_idx)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    return device


if __name__ == "__main__":
    print(get_device())
