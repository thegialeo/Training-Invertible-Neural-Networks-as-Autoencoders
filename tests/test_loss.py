import torch

from src.loss import l1_loss, l2_loss


def test_l1_loss():
    pred = torch.randn(4, 10)
    label = torch.randn(4, 10)
    loss = l1_loss(pred, label)
    assert isinstance(loss, torch.Tensor)
    assert isinstance(loss.dtype, type(torch.float32))
    assert loss >= 0


def test_l2_loss():
    pred = torch.randn(8, 10)
    label = torch.randn(8, 10)
    loss = l2_loss(pred, label)
    assert isinstance(loss, torch.Tensor)
    assert isinstance(loss.dtype, type(torch.float32))
    assert loss >= 0
