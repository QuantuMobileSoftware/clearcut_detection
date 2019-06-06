import torch
from catalyst.dl.metrics import dice
from torch.nn import Module
from torch.nn import functional as F


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
    def __init__(self, beta=0.95, reduce=True):
        super(SoftBootstrappingLoss, self).__init__()
        self.beta = beta
        self.reduce = reduce

    def forward(self, input, target):
        target_tensor = target
        prediction_tensor = input

        epsilon = 1e-7

        prediction_tensor = torch.clamp(prediction_tensor, min=epsilon, max=1 - epsilon)

        pred_tensor = torch.log(prediction_tensor / (
                    torch.ones(prediction_tensor.shape, device=torch.device('cuda:0')) - prediction_tensor))

        bootstrap_target_tensor = self.beta * target_tensor + (1.0 - self.beta) * torch.nn.Sigmoid()(pred_tensor)

        return torch.mean(F.binary_cross_entropy_with_logits(input=pred_tensor, target=bootstrap_target_tensor))