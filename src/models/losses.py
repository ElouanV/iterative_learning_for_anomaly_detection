import torch
from torch import nn
from torch.nn import functional as F


class WMSELoss(nn.Module):
    def __init__(self):
        super(WMSELoss, self).__init__()

    def forward(self, x, target, weights):
        if weights is None:
            return torch.mean((x - target) ** 2)
        return weights @ torch.mean((x - target) ** 2, dim=1) / weights.sum()


class WCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(WCrossEntropyLoss, self).__init__()

    def forward(self, x, target, weights):
        return F.cross_entropy(x, target, weight=weights)
