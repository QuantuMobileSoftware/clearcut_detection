import argparse
import os

import numpy as np
import pandas as pd
from PIL import Image
import cv2 as cv
import torch
import torchvision.transforms as transforms
from tqdm import tqdm

from torch import nn
from utils import get_filepath, count_channels, read_tensor, filter_by_channels
from models.utils import get_model


def predict(
    data_path, model_weights_path, network,
    test_df_path, save_path, size, channels
):
    model = get_model(network)
    model.encoder.conv1 = nn.Conv2d(
        count_channels(args.channels), 64, kernel_size=(7, 7),
        stride=(2, 2), padding=(3, 3), bias=False
    )

    checkpoint = torch.load(model_weights_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])

    test_df = pd.read_csv(test_df_path)

    predictions_path = os.path.join(save_path, "predictions")

    if not os.path.exists(predictions_path):
        os.makedirs(predictions_path, exist_ok=True)
        print("Prediction directory created.")

    for _, image_info in tqdm(test_df.iterrows()):
        filename = '_'.join([image_info['name'], image_info['position']])
        image_path = get_filepath(
            data_path, image_info['dataset_folder'],
            'images', filename,
            file_type='tiff'
        )

        image_tensor = filter_by_channels(
            read_tensor(image_path),
            channels
        )
        if image_tensor.ndim == 2:
            image_tensor = np.expand_dims(image_tensor, -1)

        image = transforms.ToTensor()(image_tensor)

        prediction = model.predict(image.view(1, count_channels(channels), size, size))

        result = prediction.view(size, size).detach().numpy()

        cv.imwrite(
            get_filepath(predictions_path, filename, file_type='png'),
            result * 255
        )


def image_labeling(model, unlabeled_data, image_name, img_size, channels_number=3):
    image_path = os.path.join(unlabeled_data, image_name)

    img = Image.open(image_path)

    img_tensor = transforms.ToTensor()(img)

    prediction = model.predict(img_tensor.view(1, channels_number, img_size, img_size).cuda())

    result = prediction.view(img_size, img_size).detach().cpu().numpy()
    return result


def parse_args():
    parser = argparse.ArgumentParser(
        description='Script for making predictions on test images of dataset.')
    parser.add_argument('--network', '-n', default='unet50')
    parser.add_argument('--data_path', '-dp', required=True, help='Path to directory with datasets')
    parser.add_argument('--model_weights_path', '-mwp', required=True, help='Path to file with model weights')
    parser.add_argument('--test_df', '-td', required=True, help='Path to test dataframe')
    parser.add_argument('--save_path', '-sp', required=True, help='Path to save predictions')
    parser.add_argument('--size', '-s', default=224, type=int, help='Image size')
    parser.add_argument('--channels', '-ch', default=['rgb', 'ndvi', 'ndvi_color'], nargs='+', help='Channels list')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    predict(
        args.data_path, args.model_weights_path,
        args.network, args.test_df, args.save_path,
        args.size, args.channels
    )
