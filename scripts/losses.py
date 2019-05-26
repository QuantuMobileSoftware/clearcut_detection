import torch
import torch.nn as nn
from catalyst.dl.metrics import dice
from torch.nn import functional as F


class BCE_Dice_Loss(torch.nn.Module):

    def __init__(self, bce_weight=0.5):
        super(BCE_Dice_Loss, self).__init__()
        self.bce_weight = bce_weight

    def forward(self, x, y):
        bce = F.binary_cross_entropy_with_logits(x, y)

        dice = dice_loss(x, y)

        loss = bce * self.bce_weight + dice * (1 - self.bce_weight)

        return loss


def dice_loss(pred, target, eps=1.):
    return 1 - dice(pred, target, eps)
