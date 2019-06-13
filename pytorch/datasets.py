import sys
sys.path.append('..')

import os
import imageio
import collections
import numpy as np

from catalyst.dl.utils import UtilsFactory
from params import args
from albumentations import (
    CLAHE, RandomRotate90, Flip, OneOf, Compose, RGBShift, RandomSizedCrop)
from albumentations.pytorch.transforms import ToTensor

def count_channels(channels):
    count = 0
    for ch in channels:
        if ch == 'rgb':
            count += 3
        elif ch == 'ndvi':
            count += 1
        elif ch == 'ndvi_color':
            count += 4
        elif ch == 'b2':
            count += 1
        else:
            raise Exception('{} channel is unknown!'.format(ch))

    return count


def get_fullname(*name_parts):
    return '_'.join(tuple(map(str, name_parts)))


def join_pathes(*pathes):
    return os.path.join(*pathes)


def get_filepath(*path_parts, file_type):
    return '{}.{}'.format(join_pathes(*path_parts), file_type)


def read_tensor(filepath):
    return imageio.imread(filepath)


def get_input_pair(
    data_info_row, channels=args.channels, data_path=args.data_path,
    image_folder=args.images_folder, mask_folder=args.masks_folder,
    image_type=args.image_type, mask_type=args.mask_type
):
    if len(channels) == 0:
        raise Exception('You have to specify at least one channel.')

    image_tensors = []
    for channel in channels:
        dataset = get_fullname(
            data_info_row['date'],
            data_info_row['name'],
            channel
        )
        filename = get_fullname(
            data_info_row['date'], data_info_row['name'],
            channel, data_info_row['ix'], data_info_row['iy']
        )
        image_path = get_filepath(
            data_path,
            dataset,
            image_folder,
            filename,
            file_type=image_type
        )

        image_tensor = read_tensor(image_path)
        if image_tensor.ndim == 2:
            image_tensor = image_tensor.reshape(*image_tensor.shape, 1)

        image_tensor = image_tensor / image_tensor.max() * 255
        image_tensors.append(image_tensor.astype(np.uint8))

    mask_path = get_filepath(
        data_path,
        dataset,
        mask_folder,
        filename,
        file_type=mask_type
    )

    rgb_tensor = image_tensors[0]
    masks_array = read_tensor(mask_path)

    rgb_aug = Compose([
        OneOf([
            RGBShift(),
            CLAHE(clip_limit=2)
        ], p=0.4)
    ], p=0.9)

    aug = Compose([
        RandomRotate90(),
        Flip(),
        OneOf([
            RandomSizedCrop(
                min_max_height=(int(args.image_size * 0.7), args.image_size),
                height=args.image_size, width=args.image_size)
        ], p=0.4),
        ToTensor()
    ])

    augmented_rgb = rgb_aug(image=rgb_tensor, mask=masks_array)
    images_array = np.concatenate([augmented_rgb['image'], *image_tensors[1:]], axis=2)
    augmented = aug(image=images_array, mask=augmented_rgb['mask'])
    augmented_images = augmented['image']
    augmented_masks = augmented['mask']

    return {'features': augmented_images, 'targets': augmented_masks}


def create_loaders(train_df, val_df, test_df):
    train_loader = UtilsFactory.create_loader(
        train_df,
        open_fn=get_input_pair,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True)

    valid_loader = UtilsFactory.create_loader(
        val_df,
        open_fn=get_input_pair,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True)

    test_loader = UtilsFactory.create_loader(
        test_df,
        open_fn=get_input_pair,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True)

    loaders = collections.OrderedDict()
    loaders['train'] = train_loader
    loaders['valid'] = valid_loader
    loaders['test'] = test_loader

    return loaders
