import torch

from src.loss import inn_loss, l1_loss, l2_loss, mmd_multiscale


def test_l1_loss():
    pred = torch.randn(4, 10)
    label = torch.randn(4, 10)
    loss = l1_loss(pred, label)
    assert isinstance(loss, torch.Tensor)
    assert isinstance(loss.dtype, type(torch.float32))
    assert loss.dim() == 0
    assert loss >= 0


def test_l2_loss():
    pred = torch.randn(4, 10)
    label = torch.randn(4, 10)
    loss = l2_loss(pred, label)
    assert isinstance(loss, torch.Tensor)
    assert isinstance(loss.dtype, type(torch.float32))
    assert loss.dim() == 0
    assert loss >= 0


def test_mmd_multiscale():
    p_samples = torch.randn(4, 10)
    q_samples = torch.randn(4, 10)
    loss = mmd_multiscale(p_samples, q_samples)
    assert isinstance(loss, torch.Tensor)
    assert isinstance(loss.dtype, type(torch.float32))
    assert loss.dim() == 0
    assert loss >= 0


def test_inn_loss():
    inp = torch.randn(4, 10)
    rec = torch.randn(4, 10)
    lat = torch.randn(4, 10)
    hyp_dict = {"lat_dim": 5, "a_rec": 1, "a_dist": 1, "a_sparse": 1}
    loss = inn_loss(inp, rec, lat, hyp_dict)
    assert isinstance(loss, list)
    for item in loss:
        assert isinstance(item, torch.Tensor)
        assert isinstance(item.dtype, type(torch.float32))
        assert item.dim() == 0
        assert item >= 0
