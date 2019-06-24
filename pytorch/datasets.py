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


def get_fullname(*name_parts):
    return '_'.join(tuple(map(str, name_parts)))


def join_pathes(*pathes):
    return os.path.join(*pathes)


def get_filepath(*path_parts, file_type):
    return '{}.{}'.format(join_pathes(*path_parts), file_type)


def read_tensor(filepath):
    return imageio.imread(filepath)


def scale(tensor, axis):
    for i in range(tensor.shape[axis]):
        max_value = tensor[:, :, i].max()
        if max_value != 0:
            tensor[:, :, i] = tensor[:, :, i] / max_value

    return tensor * 255


def get_input_pair(
    data_info_row, channels=args.channels,
    data_path=args.data_path, input_folder=args.input_folder,
    image_folder=args.images_folder, mask_folder=args.masks_folder,
    image_type=args.image_type, mask_type=args.mask_type
):
    if len(channels) == 0:
        raise Exception('You have to specify at least one channel.')

    year = str(data_info_row['date'])[:4]
    dataset = get_fullname(
        data_info_row['date'],
        data_info_row['name']
    )
    filename = get_fullname(
        data_info_row['date'], data_info_row['name'],
        data_info_row['ix'], data_info_row['iy']
    )
    image_path = get_filepath(
        data_path, input_folder, year,
        dataset, image_folder, filename,
        file_type=image_type
    )
    mask_path = get_filepath(
        data_path, input_folder, year,
        dataset, mask_folder, filename,
        file_type=mask_type
    )

    image_tensor = filter_by_channels(
        read_tensor(image_path),
        channels
    )

    if image_tensor.ndim == 2:
        image_tensor = image_tensor.reshape(*image_tensor.shape, 1)

    images_array = scale(image_tensor, 2).astype(np.uint16)
    masks_array = read_tensor(mask_path)

    if channels[0] == 'rgb':
        rgb_tensor = images_array[:, :, :3].astype(np.uint8)

        rgb_aug = Compose([
            OneOf([
                RGBShift(),
                CLAHE(clip_limit=2)
            ], p=0.4)
        ], p=0.9)

        augmented_rgb = rgb_aug(image=rgb_tensor, mask=masks_array)
        images_array = np.concatenate([augmented_rgb['image'], images_array[:, :, 3:]], axis=2)
        masks_array = augmented_rgb['mask']

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

    augmented = aug(image=images_array, mask=masks_array)
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
