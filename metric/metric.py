import torch
import torch.nn as nn


class Metric(nn.Module):
    def __init__(self):
        super(Metric, self).__init__()
        self.metric = None

    def forward(self, pred, target):
        return self.metric(pred, target)


class DiceMetric(nn.Module):
    def __init__(self):
        super(DiceMetric, self).__init__()

    def forward(self, pred, target):
        """calc dice

        Args:
            pred (torch.tensor): (N, H, W)
            target (torch.tensor): (N, H, W)

        Returns:
            (torch.tensor): dice
        """

        pred = pred.float()
        target = target.float()
        smooth = 1e-4

        p = torch.sigmoid(pred) > 0.5
        t = target > 0.5

        inter = (t*p).sum(dim=2).sum(dim=1).float()
        dim1 = (p).sum(dim=2).sum(dim=1).float()
        dim2 = (t).sum(dim=2).sum(dim=1).float()

        coeff = (2 * inter + smooth) / (dim1 + dim2 + smooth)
        dice_total = 1-coeff.sum(dim=0)/coeff.size(0)
        return dice_total
