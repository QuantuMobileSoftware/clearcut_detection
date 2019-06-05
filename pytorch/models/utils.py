from .resnet34_unet import ResNet34Unet
from .resnet50_unet import ResNet50Unet
from .vgg11_unet import UNet11
from .linknet import LinkNet


def get_model(name='resnet34'):
    if name == 'resnet34':
        return ResNet34Unet()
    elif name == 'resnet50':
        return ResNet50Unet()
    elif name == 'linknet':
        return LinkNet()
    elif name == 'vgg11':
        return UNet11()
