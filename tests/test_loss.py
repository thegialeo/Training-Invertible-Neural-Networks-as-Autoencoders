import torch

from src.loss import l1_loss


def test_l1_loss():
    pred = torch.randn(32, 10)
    label = torch.randn(32, 10)
    loss = l1_loss(pred, label, torch.float)
    print(type(loss))
    assert isinstance(loss, torch.Tensor)
    assert loss >= 0
