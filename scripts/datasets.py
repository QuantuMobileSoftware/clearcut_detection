import collections
import os
import pandas as pd
import numpy as np
from PIL import Image
from albumentations import (
    CLAHE, RandomRotate90, Flip, OneOf, Compose, RandomSizedCrop, RGBShift
)
from albumentations.pytorch.transforms import ToTensor
from catalyst.dl.utils import UtilsFactory

from params import args


def get_image(image_info):
    dataset_path = args.dataset_path
    img_path = os.path.join(dataset_path, image_info["dataset_folder"], "images", image_info["image_name"] + ".tiff")
    mask_path = os.path.join(dataset_path, image_info["dataset_folder"], "masks", image_info["image_name"] + ".png")

    img = Image.open(img_path)
    mask = Image.open(mask_path)

    img_array = np.array(img)
    mask_array = np.array(mask)

    aug = Compose([
        RandomRotate90(),
        Flip(),
        OneOf([
            RGBShift(),
            CLAHE(clip_limit=2)
        ], p=0.4),
        RandomSizedCrop(min_max_height=(int(args.img_height * 0.7), args.img_height), height=args.img_height,
                        width=args.img_width),
        ToTensor()
    ], p=0.9)

    augmented = aug(image=img_array, mask=mask_array)

    augmented_img = augmented['image']
    augmented_mask = augmented['mask']

    return {"features": augmented_img, "targets": augmented_mask}


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
