import segmentation_models_pytorch as smp

from .linknet import LinkNet


def get_model(name='unet34'):
    if name == 'unet34':
        return smp.Unet('resnet34', encoder_weights='imagenet')
    elif name == 'unet50':
        return smp.Unet('resnet50', encoder_weights='imagenet')
    elif name == 'linknet':
        return LinkNet()
    # elif name == 'linknet34':
    #     return smp.Linknet('resnet34', encoder_weights='imagenet')
    # elif name == 'linknet50':
    #     return smp.Linknet('resnet50', encoder_weights='imagenet')
    else:
        raise ValueError("Unknown network")
