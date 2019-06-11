import segmentation_models_pytorch as smp
import torch
import torchvision.models as models
from segmentation_models_pytorch.encoders.resnet import ResNetEncoder, resnet_encoders
from params import args


def get_satelite_pretrained_resnet(name='resnet50', pretrained=True):
    Encoder = resnet_encoders[name]['encoder']
    encoder = Encoder(**resnet_encoders[name]['params'])
    encoder.out_shapes = resnet_encoders[name]['out_shapes']

    if pretrained:
        checkpoint = torch.load(args.weights_path)
        encoder.load_state_dict(checkpoint)

    return encoder


def get_model(name='unet34'):
    if name == 'unet34':
        return smp.Unet('resnet34', encoder_weights='imagenet')
    elif name == 'unet50':
        return smp.Unet('resnet50', encoder_weights='imagenet')
    elif name == 'unet101':
        return smp.Unet('resnet101', encoder_weights='imagenet')
    elif name == 'linknet34':
        return smp.Linknet('resnet34', encoder_weights='imagenet')
    elif name == 'linknet50':
        return smp.Linknet('resnet50', encoder_weights='imagenet')
    elif name == 'fpn34':
        return smp.FPN('resnet34', encoder_weights='imagenet')
    elif name == 'fpn50':
        return smp.FPN('resnet50', encoder_weights='imagenet')
    elif name == 'fpn50_satelite':
        fpn_resnet50 = smp.FPN('resnet50', encoder_weights='imagenet')
        fpn_resnet50.encoder = get_satelite_pretrained_resnet()
        return fpn_resnet50
    elif name == 'fpn101':
        return smp.FPN('resnet101', encoder_weights='imagenet')
    elif name == 'pspnet34':
        return smp.PSPNet('resnet34', encoder_weights='imagenet')
    elif name == 'pspnet50':
        return smp.PSPNet('resnet50', encoder_weights='imagenet')
    else:
        raise ValueError("Unknown network")
