import collections
import numpy as np
import pandas as pd

from catalyst.dl.utils import UtilsFactory
from albumentations import (
    CLAHE, RandomRotate90, Flip, OneOf, Compose, RGBShift, RandomSizedCrop, RandomCrop)
from albumentations.pytorch.transforms import ToTensor

from utils import get_filepath, read_tensor, filter_by_channels
from params import args


def add_record(data_info, dataset_folder, name, position):
    return data_info.append(
        pd.DataFrame({
            'dataset_folder': dataset_folder,
            'name': name,
            'position': position
        }, index=[0]),
        sort=True, ignore_index=True
    )


def get_input_pair(
    data_info_row, channels=args.channels, data_path=args.dataset_path,
    images_folder="images", image_type="tiff", masks_folder="masks", mask_type="png"
):
    if len(channels) == 0:
        raise Exception('You have to specify at least one channel.')

    instance_name = '_'.join([data_info_row['name'], data_info_row['position']])
    image_path = get_filepath(
        data_path, data_info_row['dataset_folder'], images_folder,
        instance_name, file_type=image_type
    )
    mask_path = get_filepath(
        data_path,  data_info_row['dataset_folder'], masks_folder,
        instance_name, file_type=mask_type
    )

    images_array = filter_by_channels(
        read_tensor(image_path),
        channels
    )

    if images_array.ndim == 2:
        images_array = np.expand_dims(images_array, -1)

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
        RandomCrop(224,224),
        RandomRotate90(),
        Flip(),
        # OneOf([
        #     RandomSizedCrop(
        #         min_max_height=(int(args.image_size * 0.7), args.image_size),
        #         height=args.image_size, width=args.image_size)
        # ], p=0.4),
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
