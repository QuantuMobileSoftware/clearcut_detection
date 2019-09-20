import os

import numpy as np
from PIL import Image
from albumentations import (
    CLAHE, RandomRotate90, Flip, OneOf, Compose, RGBShift, RandomCrop
)
from albumentations.pytorch.transforms import ToTensor

from clearcut_research.pytorch.dataset import Dataset


class AutoencoderDataset(Dataset):
    def get_input_pair(self, image_info):
        dataset_path = self.dataset_path
        img_path = os.path.join(dataset_path, image_info["dataset_folder"], self.images_folder,
                                image_info["name"] + '_' + image_info["position"] + '.' + self.image_type)

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
