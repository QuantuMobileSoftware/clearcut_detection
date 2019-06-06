import sys
sys.path.append('..')

import torch
import collections
import numpy as np

from torchvision import transforms
from catalyst.dl.utils import UtilsFactory
from preprocessing.generate_data import get_filepath, get_fullname, read_tensor
from params import args


def get_input_pair(
    data_info_row, channels=args.channels, data_path=args.data_path,
    image_folder=args.images_folder, mask_folder=args.masks_folder,
    image_type=args.image_type, mask_type=args.mask_type
):

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

    image = transforms.ToTensor()(np.concatenate(image_tensors, axis=2))
    mask = transforms.ToTensor()(read_tensor(mask_path))

    return {'features': image, 'targets': mask}


def create_loaders(train_df, val_df):
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

    loaders = collections.OrderedDict()
    loaders["train"] = train_loader
    loaders["valid"] = valid_loader

    return loaders