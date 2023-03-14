import pytest
import torch

from src.functionalities import get_device
from src.loss import LossTracker


@pytest.mark.parametrize("INN", [True, False])
def test_losstracker_compute(INN):
    hyp_dict = {
        "num_epoch": 3,
        "a_rec": 1,
        "a_dist": 1,
        "a_sparse": 1,
        "dtype": torch.float32,
        "INN": INN,
    }
    loss_tracker = LossTracker(5, hyp_dict, device=get_device())
    pred = torch.randn(4, 10)
    label = torch.randn(4, 10)
    lat = torch.randn(4, 10)
    l1_loss = loss_tracker.l1_loss(pred, label)
    l2_loss = loss_tracker.l2_loss(pred, label)
    mmd_loss = loss_tracker.mmd_multiscale(pred, label)
    inn_loss = loss_tracker.inn_loss(pred, label, lat)
    assert isinstance(l1_loss, torch.Tensor)
    assert isinstance(l1_loss.dtype, type(torch.float32))
    assert l1_loss.dim() == 0
    assert l1_loss >= 0
    assert isinstance(l2_loss, torch.Tensor)
    assert isinstance(l2_loss.dtype, type(torch.float32))
    assert l2_loss.dim() == 0
    assert l2_loss >= 0
    assert isinstance(mmd_loss, torch.Tensor)
    assert isinstance(mmd_loss.dtype, type(torch.float32))
    assert mmd_loss.dim() == 0
    assert mmd_loss >= 0
    assert isinstance(inn_loss, list)
    assert len(inn_loss) == 4
    for item in inn_loss:
        assert isinstance(item, torch.Tensor)
        assert isinstance(item.dtype, type(torch.float32))
        assert item.dim() == 0
        assert item >= 0


def test_losstracker_update_inn():
    hyp_dict = {
        "num_epoch": 3,
        "a_rec": 1,
        "a_dist": 1,
        "a_sparse": 1,
        "dtype": torch.float32,
        "INN": True,
    }
    loss_tracker = LossTracker(5, hyp_dict, device=get_device())
    train_losses = []
    test_losses = []
    for i in range(3):
        train_loss_lst = loss_tracker.inn_loss(
            torch.randn(4, 10), torch.randn(4, 10), torch.randn(4, 10)
        )
        loss_tracker.update_inn_loss(train_loss_lst, i, mode="train")
        train_losses.append(train_loss_lst)
        test_loss_lst = loss_tracker.inn_loss(
            torch.randn(4, 10), torch.randn(4, 10), torch.randn(4, 10)
        )
        loss_tracker.update_inn_loss(test_loss_lst, i, mode="test")
        test_losses.append(test_loss_lst)
    recorded_train_loss = loss_tracker.get_loss(mode="train")
    recorded_test_loss = loss_tracker.get_loss(mode="test")
    assert isinstance(recorded_train_loss, dict)
    assert isinstance(recorded_test_loss, dict)
    assert isinstance(recorded_train_loss["total"], torch.Tensor)
    assert isinstance(recorded_train_loss["rec"], torch.Tensor)
    assert isinstance(recorded_train_loss["dist"], torch.Tensor)
    assert isinstance(recorded_train_loss["sparse"], torch.Tensor)
    assert isinstance(recorded_train_loss["total"].dtype, type(torch.float32))
    assert isinstance(recorded_train_loss["rec"].dtype, type(torch.float32))
    assert isinstance(recorded_train_loss["dist"].dtype, type(torch.float32))
    assert isinstance(recorded_train_loss["sparse"].dtype, type(torch.float32))
    assert isinstance(recorded_test_loss["total"], torch.Tensor)
    assert isinstance(recorded_test_loss["rec"], torch.Tensor)
    assert isinstance(recorded_test_loss["dist"], torch.Tensor)
    assert isinstance(recorded_test_loss["sparse"], torch.Tensor)
    assert isinstance(recorded_test_loss["total"].dtype, type(torch.float32))
    assert isinstance(recorded_test_loss["rec"].dtype, type(torch.float32))
    assert isinstance(recorded_test_loss["dist"].dtype, type(torch.float32))
    assert isinstance(recorded_test_loss["sparse"].dtype, type(torch.float32))
    assert len(recorded_train_loss) == 4
    assert len(recorded_test_loss) == 4
    assert recorded_train_loss["total"].size() == (3,)
    assert recorded_train_loss["rec"].size() == (3,)
    assert recorded_train_loss["dist"].size() == (3,)
    assert recorded_train_loss["sparse"].size() == (3,)
    assert recorded_test_loss["total"].size() == (3,)
    assert recorded_test_loss["rec"].size() == (3,)
    assert recorded_test_loss["dist"].size() == (3,)
    assert recorded_test_loss["sparse"].size() == (3,)
    for i, target in enumerate(train_losses):
        assert isinstance(recorded_train_loss["total"][i], torch.Tensor)
        assert isinstance(recorded_train_loss["rec"][i], torch.Tensor)
        assert isinstance(recorded_train_loss["dist"][i], torch.Tensor)
        assert isinstance(recorded_train_loss["sparse"][i], torch.Tensor)
        assert isinstance(recorded_train_loss["total"][i].dtype, type(torch.float32))
        assert isinstance(recorded_train_loss["rec"][i].dtype, type(torch.float32))
        assert isinstance(recorded_train_loss["dist"][i].dtype, type(torch.float32))
        assert isinstance(recorded_train_loss["sparse"][i].dtype, type(torch.float32))
        assert recorded_train_loss["total"][i].dim() == 0
        assert recorded_train_loss["rec"][i].dim() == 0
        assert recorded_train_loss["dist"][i].dim() == 0
        assert recorded_train_loss["sparse"][i].dim() == 0
        assert recorded_train_loss["total"][i] == target[0]
        assert recorded_train_loss["rec"][i] == target[1]
        assert recorded_train_loss["dist"][i] == target[2]
        assert recorded_train_loss["sparse"][i] == target[3]
    for i, target in enumerate(test_losses):
        assert isinstance(recorded_test_loss["total"][i], torch.Tensor)
        assert isinstance(recorded_test_loss["rec"][i], torch.Tensor)
        assert isinstance(recorded_test_loss["dist"][i], torch.Tensor)
        assert isinstance(recorded_test_loss["sparse"][i], torch.Tensor)
        assert isinstance(recorded_test_loss["total"][i].dtype, type(torch.float32))
        assert isinstance(recorded_test_loss["rec"][i].dtype, type(torch.float32))
        assert isinstance(recorded_test_loss["dist"][i].dtype, type(torch.float32))
        assert isinstance(recorded_test_loss["sparse"][i].dtype, type(torch.float32))
        assert recorded_test_loss["total"][i].dim() == 0
        assert recorded_test_loss["rec"][i].dim() == 0
        assert recorded_test_loss["dist"][i].dim() == 0
        assert recorded_test_loss["sparse"][i].dim() == 0
        assert recorded_test_loss["total"][i] == target[0]
        assert recorded_test_loss["rec"][i] == target[1]
        assert recorded_test_loss["dist"][i] == target[2]
        assert recorded_test_loss["sparse"][i] == target[3]


def test_losstracker_update_classic():
    hyp_dict = {
        "num_epoch": 3,
        "a_rec": 1,
        "a_dist": 1,
        "a_sparse": 1,
        "dtype": torch.float32,
        "INN": False,
    }
    loss_tracker = LossTracker(5, hyp_dict, device=get_device())
    train_losses = []
    test_losses = []
    for i in range(3):
        loss_train = loss_tracker.l1_loss(torch.randn(4, 10), torch.randn(4, 10))
        loss_tracker.update_classic_loss(loss_train, i, mode="train")
        train_losses.append(loss_train)
        loss_test = loss_tracker.l1_loss(torch.randn(4, 10), torch.randn(4, 10))
        loss_tracker.update_classic_loss(loss_test, i, mode="test")
        test_losses.append(loss_test)
    recorded_train_loss = loss_tracker.get_loss(mode="train")
    recorded_test_loss = loss_tracker.get_loss(mode="test")
    assert isinstance(recorded_train_loss, dict)
    assert isinstance(recorded_test_loss, dict)
    assert isinstance(recorded_train_loss["rec"], torch.Tensor)
    assert isinstance(recorded_train_loss["rec"].dtype, type(torch.float32))
    assert isinstance(recorded_test_loss["rec"], torch.Tensor)
    assert isinstance(recorded_test_loss["rec"].dtype, type(torch.float32))
    assert len(recorded_train_loss) == 1
    assert len(recorded_test_loss) == 1
    assert recorded_train_loss["rec"].size() == (3,)
    assert recorded_test_loss["rec"].size() == (3,)
    for i, target in enumerate(train_losses):
        assert isinstance(recorded_train_loss["rec"][i], torch.Tensor)
        assert isinstance(recorded_train_loss["rec"][i].dtype, type(torch.float))
        assert recorded_train_loss["rec"][i].dim() == 0
        assert recorded_train_loss["rec"][i] == target
    for i, target in enumerate(test_losses):
        assert isinstance(recorded_test_loss["rec"][i], torch.Tensor)
        assert isinstance(recorded_test_loss["rec"][i].dtype, type(torch.float))
        assert recorded_test_loss["rec"][i].dim() == 0
        assert recorded_test_loss["rec"][i] == target


@pytest.mark.parametrize("INN", [True, False])
def test_losstracker_raises(INN):
    hyp_dict = {
        "num_epoch": 3,
        "a_rec": 1,
        "a_dist": 1,
        "a_sparse": 1,
        "dtype": torch.float32,
        "INN": INN,
    }
    loss_tracker = LossTracker(5, hyp_dict, device=get_device())
    l1_loss = loss_tracker.l1_loss(torch.randn(4, 10), torch.randn(4, 10))
    inn_loss = loss_tracker.inn_loss(
        torch.randn(4, 10), torch.randn(4, 10), torch.randn(4, 10)
    )
    with pytest.raises(ValueError):
        loss_tracker.update_classic_loss(l1_loss, 0, mode="non-valid-mode")
    with pytest.raises(ValueError):
        loss_tracker.update_inn_loss(inn_loss, 0, mode="non-valid-mode")
    with pytest.raises(ValueError):
        loss_tracker.get_loss(mode="non-valid-mode")
