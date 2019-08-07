import argparse
import os

import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
from PIL import Image
from torchvision.transforms import ToTensor
from tqdm import tqdm

from clearcut_research.pytorch.metrics import multi_class_dice


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
    parser.add_argument(
        '--image_size', '-is', type=int,
        required=True, help='Image size')
    parser.add_argument(
        '--classes', '-c', type=int, default=3,
        help='Number of classes')
    return parser.parse_args()


def evaluate(datasets_path, test_df_path, model_weights_path, image_size, classes, images_folder="images",
             image_type="tiff", masks_folder="masks", mask_type="png"):
    filenames = pd.read_csv(test_df_path)
    model = smp.FPN('resnet50', encoder_weights='imagenet', classes=classes, activation='softmax')
    checkpoint = torch.load(model_weights_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    dices = []

    for ind, image_info in tqdm(filenames.iterrows()):
        name = image_info["name"] + '_' + image_info["position"]

        image = Image.open(os.path.join(datasets_path, image_info["dataset_folder"], images_folder,
                                        name + '.' + image_type))
        mask = Image.open(os.path.join(datasets_path, image_info["dataset_folder"], masks_folder,
                                       name + '.' + mask_type))

        img_array = np.array(image)
        mask_array = np.array(mask).astype(np.float32)

        image_tensor = ToTensor()(img_array)
        mask_tensor = ToTensor()(mask_array)

        prediction = model.forward(image_tensor.view(1, classes, image_size, image_size))
        mask_tensor = mask_tensor.squeeze().view(1, image_size, image_size).long()

        dice_score = multi_class_dice(prediction, mask_tensor)
        dices.append(dice_score.item())

    print("Average dice score - {0}".format(round(np.average(dices), 4)))


if __name__ == "__main__":
    args = parse_args()
    evaluate(args.datasets_path, args.test_df_path, args.model_weights_path, args.image_size, args.classes)
