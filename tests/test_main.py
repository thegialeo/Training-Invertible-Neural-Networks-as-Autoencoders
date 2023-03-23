import random
import string

import pytest
from docopt import DocoptExit

from src.main import main


@pytest.mark.parametrize("dataset", ["mnist", "cifar", "celeba"])
def test_main_valid_input(dataset, capsys):
    args = main([dataset, "--test_mode"])
    _, err = capsys.readouterr()
    assert isinstance(err, str)
    assert isinstance(args, dict)
    assert err == ""
    assert len(args) == 4
    for key, value in args.items():
        if key == dataset or key == "--test_mode":
            assert bool(value)
        else:
            assert not bool(value)


def test_main_invalid_input():
    random_input = "".join(random.choices(string.ascii_letters, k=random.randrange(10)))
    with pytest.raises(DocoptExit):
        main([random_input])
