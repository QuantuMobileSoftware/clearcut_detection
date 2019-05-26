def dice_loss(prediction, target, smooth=1.):
    prediction = prediction.contiguous()
    target = target.contiguous()

    intersection = (prediction * target).sum(dim=2).sum(dim=2)

    loss = (1 - ((2. * intersection + smooth) / (prediction.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))

    return loss.mean()
