import collections
import os

import numpy as np
import pandas as pd
from PIL import Image
from albumentations import (
    CLAHE, RandomRotate90, Flip, OneOf, Compose, RGBShift, RandomCrop, RandomSizedCrop
)
from albumentations.pytorch.transforms import ToTensor
from catalyst.dl.utils import UtilsFactory

from params import args


def get_image(image_info, images_folder="images", image_type="tiff"):
    dataset_path = args.dataset_path
    img_path = os.path.join(dataset_path, image_info["dataset_folder"], images_folder,
                            image_info["name"] + '_' + image_info["position"] + '.' + image_type)

    img = Image.open(img_path)

    img_array = np.array(img)

    augm = Compose([
        RandomCrop(224, 224),
        RandomRotate90(),
        Flip(),
        OneOf([
            RGBShift(),
            CLAHE(clip_limit=2)
        ], p=0.4),
        ToTensor()
    ], p=1)

    augmented = augm(image=img_array)

    augmented_img = augmented['image']

    return {"features": augmented_img, "targets": augmented_img}


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
