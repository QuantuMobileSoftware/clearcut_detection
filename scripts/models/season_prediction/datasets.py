import collections
import datetime
import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from albumentations import (
    CLAHE, RandomRotate90, Flip, OneOf, Compose, RGBShift, RandomCrop
)
from albumentations.pytorch.transforms import ToTensor
from catalyst.dl.utils import UtilsFactory

from params import args


def get_image(image_info):
    dataset_path = args.dataset_path
    img_path = os.path.join(dataset_path, image_info["dataset_folder"], "images",
                            image_info["name"] + '_' + image_info["channel"] + '_' + image_info["position"] + '.' +
                            image_info["image_type"])
    mask_path = os.path.join(dataset_path, image_info["dataset_folder"], "masks",
                             image_info["name"] + '_' + image_info["channel"] + '_' + image_info["position"] + '.' +
                             image_info["mask_type"])

    img = Image.open(img_path)
    mask = Image.open(mask_path)

    img_array = np.array(img)
    mask_array = np.array(mask)

    aug = Compose([
        RandomCrop(224, 224),
        RandomRotate90(),
        Flip(),
        OneOf([
            RGBShift(),
            CLAHE(clip_limit=2)
        ], p=0.4),
        # OneOf([
        #     RandomSizedCrop(min_max_height=(int(image_info['image_size'] * 0.7), image_info['image_size']),
        #                     height=image_info['image_size'],
        #                     width=image_info['image_size'])
        # ], p=0.4),
        ToTensor()
    ], p=1)

    augmented = aug(image=img_array, mask=mask_array)

    augmented_img = augmented['image']
    augmented_mask = augmented['mask']

    date_time = datetime.datetime.strptime(image_info["name"].split('_')[0], '%Y%m%d')

    winter = torch.tensor([0.])
    winter_months = [12, 1, 2]

    if date_time.month in winter_months:
        winter = torch.tensor([1.])

    return {"features": augmented_img, "targets": (augmented_mask, winter)}


def create_loaders():
    train_df = pd.read_csv(args.train_df)
    val_df = pd.read_csv(args.val_df)

    train_df = train_df.to_dict('records')
    val_df = val_df.to_dict('records')

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
