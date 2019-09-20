import datetime
import os
import numpy as np
import torch
from PIL import Image
from albumentations import (
    CLAHE, RandomRotate90, Flip, OneOf, Compose, RGBShift, RandomCrop
)
from albumentations.pytorch.transforms import ToTensor

from clearcut_research.pytorch.dataset import Dataset


class SeasonDataset(Dataset):
    def get_input_pair(self, image_info):
        dataset_path = self.dataset_path
        img_path = os.path.join(dataset_path, image_info["dataset_folder"], self.images_folder,
                                image_info["name"] + '_' + image_info["position"] + '.' + self.image_type)
        mask_path = os.path.join(dataset_path, image_info["dataset_folder"], self.masks_folder,
                                 image_info["name"] + '_' + image_info["position"] + '.' + self.mask_type)

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
