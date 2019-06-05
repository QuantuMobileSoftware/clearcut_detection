import torch
from catalyst.dl.metrics import dice
from torch.nn import functional as F
from torch.nn import Module


class BCE_Dice_Loss(Module):

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


class SoftBootstrappingLoss(Module):
    """
    Loss(t, p) = - (beta * t + (1 - beta) * p) * log(p)
    """
    def __init__(self, beta=0.95, reduce=True):
        super(SoftBootstrappingLoss, self).__init__()
        self.beta = beta
        self.reduce = reduce

    def forward(self, input, target):
        # cross_entropy = - t * log(p)
        beta_xentropy = self.beta * F.cross_entropy(input, target, reduce=False)

        # second term = - (1 - beta) * p * log(p)
        bootstrap = - (1.0 - self.beta) * torch.sum(F.softmax(input, dim=1) * F.log_softmax(input, dim=1), dim=1)

        if self.reduce:
            return torch.mean(beta_xentropy + bootstrap)
        return beta_xentropy + bootstrap


class HardBootstrappingLoss(Module):
    """
    Loss(t, p) = - (beta * t + (1 - beta) * z) * log(p)
    where z = argmax(p)
    """
    def __init__(self, beta=0.8, reduce=True):
        super(HardBootstrappingLoss, self).__init__()
        self.beta = beta
        self.reduce = reduce

    def forward(self, input, target):
        # cross_entropy = - t * log(p)
        beta_xentropy = self.beta * F.cross_entropy(input, target, reduce=False)

        # z = argmax(p)
        _, z = torch.max(F.softmax(input, dim=1), dim=1)
        z = z.view(-1, 1)
        bootstrap = F.log_softmax(input, dim=1).gather(1, z).view(-1)
        # second term = (1 - beta) * z * log(p)
        bootstrap = - (1.0 - self.beta) * bootstrap

        if self.reduce:
            return torch.mean(beta_xentropy + bootstrap)
        return beta_xentropy + bootstrap