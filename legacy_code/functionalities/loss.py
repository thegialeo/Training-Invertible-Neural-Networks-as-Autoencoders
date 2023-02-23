import numpy as np
import torch


def l1_loss(x, y):
    return torch.mean(torch.abs(x - y))


def l2_loss(x, y):
    return torch.mean((x - y) ** 2)


def feat_loss(x, y, feat_model):
    return torch.mean(torch.abs(feat_model(x) - feat_model(y)))


def MMD_multiscale(x, y, device):
    x, y = x.to(device), y.to(device)
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())

    rx = xx.diag().unsqueeze(0).expand_as(xx)
    ry = yy.diag().unsqueeze(0).expand_as(yy)

    dxx = rx.t() + rx - 2.0 * xx
    dyy = ry.t() + ry - 2.0 * yy
    dxy = rx.t() + ry - 2.0 * zz

    XX, YY, XY = (
        torch.zeros(xx.shape).to(device),
        torch.zeros(xx.shape).to(device),
        torch.zeros(xx.shape).to(device),
    )

    for a in [0.2, 0.5, 0.9, 1.3]:
        XX += a**2 * (a**2 + dxx) ** -1
        YY += a**2 * (a**2 + dyy) ** -1
        XY += a**2 * (a**2 + dxy) ** -1

    return torch.mean(XX + YY - 2.0 * XY)


def shuffle(x):
    x = x.permute(1, 0).cpu()
    shape = x.size()
    rand_x = torch.FloatTensor()
    for i in range(x.size()[0]):
        idx = torch.randperm(x.size()[1])
        temp = x[i, idx]
        rand_x = torch.cat((rand_x, temp))
    rand_x = rand_x.view(shape)
    rand_x = rand_x.permute(1, 0)
    return rand_x
