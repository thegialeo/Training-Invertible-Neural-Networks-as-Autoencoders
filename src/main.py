"""Provide command line user interface.

Usage:
    main.py mnist [-t | --test_mode]
    main.py cifar [-t | --test_mode]
    main.py celeba [-t | --test_mode]

Options:
    -h, --help          Show this help message
    -t, --test_mode     Reduce runtime for unit testing
"""
from collections.abc import Sequence
from typing import Optional

from docopt import docopt
from tabulate import tabulate

from src.experiment import experiment_wrapper


def main(argv: Optional[Sequence[str]] = None) -> dict:
    """Run command line interface.

    Args:
        argv (Sequence): Command line arguments vector/list.

    Returns:
        args (dict): user input given through command line interface
    """
    args = docopt(__doc__, argv)

    print("Interface user input:")
    print(tabulate(list(args.items()), missingval="None"))

    if bool(args["mnist"]):
        dataset = "mnist"
        model_lst = [
            "mnist_inn",
            "mnist_classic",
            "mnist_classic1024",
            "mnist_classicDeep1024",
            "mnist_classic2048",
        ]
    elif bool(args["cifar"]):
        dataset = "cifar"
        model_lst = ["cifar_inn", "cifar_classic"]
    elif bool(args["celeba"]):
        dataset = "celeba"
        model_lst = ["celeba_inn", "celeba_classic"]

    experiment_wrapper(model_lst, dataset, bool(args["--test_mode"]))

    return args


if __name__ == "__main__":
    main()
