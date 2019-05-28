import collections
import os

import torchvision.transforms as transforms

from catalyst.dl.utils import UtilsFactory
from PIL import Image

from params import args


def get_image(data_info):
    filename = '_'.join([
        data_info['name'],
        data_info['channel'],
        data_info['position']])
    image_path = os.path.join(
        data_info['image_path'],
        '{}.{}'.format(
            filename,
            data_info['image_type']))
    mask_path = os.path.join(
        data_info['mask_path'],
        '{}.{}'.format(
            filename,
            data_info['mask_type']))

    img = Image.open(image_path)
    mask = Image.open(mask_path)

    img_tensor = transforms.ToTensor()(img)
    mask_tensor = transforms.ToTensor()(mask)

    return {'features': img_tensor, 'targets': mask_tensor}


def create_loaders(train_df, val_df):
    train_loader = UtilsFactory.create_loader(
        train_df,
        open_fn=get_image,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True)

    valid_loader = UtilsFactory.create_loader(
        val_df,
        open_fn=get_image,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True)

    loaders = collections.OrderedDict()
    loaders["train"] = train_loader
    loaders["valid"] = valid_loader
    return loaders
