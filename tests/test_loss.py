import pytest
import torch

from src.loss import LossTracker


@pytest.mark.parametrize("INN", [True, False])
def test_LossTracker_compute(INN):
    hyp_dict = {
        "num_epoch": 3,
        "lat_dim": 5,
        "a_rec": 1,
        "a_dist": 1,
        "a_sparse": 1,
        "dtype": torch.float32,
        "INN": INN,
    }
    loss_tracker = LossTracker(hyp_dict)
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
    for item in inn_loss:
        assert isinstance(item, torch.Tensor)
        assert isinstance(item.dtype, type(torch.float32))
        assert item.dim() == 0
        assert item >= 0


def test_LossTracker_update_classic():
    hyp_dict = {
        "num_epoch": 3,
        "lat_dim": 5,
        "a_rec": 1,
        "a_dist": 1,
        "a_sparse": 1,
        "dtype": torch.float32,
        "INN": False,
    }
    loss_tracker = LossTracker(hyp_dict)
    train_losses = []
    test_losses = []
    for i in range(3):
        loss = loss_tracker.l1_loss(torch.randn(4, 10), torch.randn(4, 10))
        loss_tracker.update_classic_loss(loss, i, mode="train")
        train_losses.append(loss)
    for i in range(3):
        loss = loss_tracker.l1_loss(torch.randn(4, 10), torch.randn(4, 10))
        loss_tracker.update_classic_loss(loss, i, mode="test")
        test_losses.append(loss)
