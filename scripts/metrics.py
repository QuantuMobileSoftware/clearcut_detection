import torch
from torch.nn import functional as F


def multi_class_dice(input, target, eps=1e-7):
    num_classes = input.shape[1]
    true_1_hot = torch.eye(num_classes)[target]
    true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
    probas = F.softmax(input, dim=1)
    true_1_hot = true_1_hot.type(input.type())
    dims = (0,) + tuple(range(2, input.ndimension()))

    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    dice_score = (2. * intersection / (cardinality + eps)).mean().item()
    return dice_score
