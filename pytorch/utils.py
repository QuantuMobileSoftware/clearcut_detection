import os
import imageio
import numpy as np
import segmentation_models_pytorch as smp


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
    elif name == 'fpn101':
        return smp.FPN('resnet101', encoder_weights='imagenet')
    elif name == 'pspnet34':
        return smp.PSPNet('resnet34', encoder_weights='imagenet')
    elif name == 'pspnet50':
        return smp.PSPNet('resnet50', encoder_weights='imagenet')
    else:
        raise ValueError("Unknown network")


def join_pathes(*pathes):
    return os.path.join(*pathes)


def get_filepath(*path_parts, file_type):
    return '{}.{}'.format(join_pathes(*path_parts), file_type)


def read_tensor(filepath):
    return imageio.imread(filepath)


def count_channels(channels):
    count = 0
    for ch in channels:
        if ch == 'rgb':
            count += 3
        elif ch == 'ndvi_color':
            count += 4
        elif ch in ['ndvi', 'b2', 'b3', 'b4', 'b8']:
            count += 1
        else:
            raise Exception('{} channel is unknown!'.format(ch))

    return count


def filter_by_channels(image_tensor, channels):
    result = []
    for ch in channels:
        if ch == 'rgb':
            result.append(image_tensor[:, :, :3])
        elif ch == 'ndvi':
            result.append(image_tensor[:, :, 3:4])
        elif ch == 'ndvi_color':
            result.append(image_tensor[:, :, 4:8])
        elif ch == 'b2':
            result.append(image_tensor[:, :, 8:9])
        elif ch == 'b3':
            result.append(image_tensor[:, :, 9:10])
        elif ch == 'b4':
            result.append(image_tensor[:, :, 10:11])
        elif ch == 'b8':
            result.append(image_tensor[:, :, 11:12])
        else:
            raise Exception('{} channel is unknown!'.format(ch))

    return np.concatenate(result, axis=2)