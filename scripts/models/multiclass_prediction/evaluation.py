import argparse
import os
import sys

import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
from PIL import Image
from albumentations import (
    Compose, RandomCrop
)
from albumentations.pytorch.transforms import ToTensor
from tqdm import tqdm

sys.path.append("../..")
from metrics import multi_class_dice


def parse_args():
    parser = argparse.ArgumentParser(
        description='Script for evaluating performance of the model.')
    parser.add_argument(
        '--datasets_path', '-dp', dest='datasets_path',
        required=True, help='Path to the directory all the data')
    parser.add_argument(
        '--test_df_path', '-tp', dest='test_df_path',
        required=True, help='Path to the test dataframe with image names')
    parser.add_argument(
        '--model_weights_path', '-mwp',
        required=True, help='Path to model weights')
    return parser.parse_args()


def evaluate(datasets_path, test_df_path, model_weights_path, images_folder="images", image_type="tiff",
             masks_folder="masks", mask_type="png"):
    filenames = pd.read_csv(test_df_path)
    model = smp.FPN('resnet50', encoder_weights='imagenet', classes=3, activation='softmax')
    checkpoint = torch.load(model_weights_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    dices = []

    for ind, image_info in tqdm(filenames.iterrows()):
        name = image_info["name"] + '_' + image_info["position"]

        image = Image.open(os.path.join(datasets_path, image_info["name"], images_folder,
                                        name + '.' + image_type))
        mask = Image.open(os.path.join(datasets_path, image_info["name"], masks_folder,
                                       name + '.' + mask_type))

        img_array = np.array(image)
        mask_array = np.array(mask).astype(np.float32)

        aug = Compose([
            RandomCrop(224, 224),
            ToTensor()
        ], p=1)

        augmented = aug(image=img_array, mask=mask_array)
        image_tensor = augmented['image']
        mask_tensor = augmented['mask']

        prediction = model.forward(image_tensor.view(1, 3, 224, 224))
        mask_tensor = mask_tensor.squeeze().view(1, 224, 224).long()

        dice_score = multi_class_dice(prediction, mask_tensor)
        dices.append(dice_score)

    print("Average dice score - {0}".format(round(np.average(dices), 4)))


if __name__ == "__main__":
    args = parse_args()
    evaluate(args.datasets_path, args.test_df_path, args.model_weights_path)
