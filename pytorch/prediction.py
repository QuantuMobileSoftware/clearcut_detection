import argparse
import os

import numpy as np
import cv2 as cv
import pandas as pd
import torch
import torchvision.transforms as transforms
from tqdm import tqdm

from torch import nn
from utils import get_model
from datasets import get_filepath, get_fullname, count_channels, read_tensor


def parse_args():
    parser = argparse.ArgumentParser(
        description='Script for making predictions on test images of dataset.')
    parser.add_argument('--network', '-n', default='fpn50')
    parser.add_argument('--data_path', '-dp', required=True, help='Path to directory with datasets')
    parser.add_argument('--model_weights_path', '-mwp', default='../logs/checkpoints/best.pth', help='Path to file with model weights')
    parser.add_argument('--test_df', '-td', default='../data/test_df.csv', help='Path to test dataframe')
    parser.add_argument('--save_path', '-sp', required=True, help='Path to save predictions')
    parser.add_argument('--size', '-s', default=224, type=int, help='Image size')
    parser.add_argument('--channels', '-ch', default=['rgb', 'ndvi', 'ndvi_color', 'b2'], type=list, help='Channels list')

    return parser.parse_args()


def predict(data_path, model_weights_path, network, test_df_path, save_path, size, channels):
    model = get_model(network)
    model.encoder.conv1 = nn.Conv2d(count_channels(args.channels), 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    checkpoint = torch.load(model_weights_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])

    test_df = pd.read_csv(test_df_path)

    predictions_path = os.path.join(save_path, "predictions")

    if not os.path.exists(predictions_path):
        os.mkdir(predictions_path)
        print("Prediction directory created.")

    for _, image_info in tqdm(test_df.iterrows()):
        image_tensors = []
        for channel in channels:
            dataset = get_fullname(
                image_info['date'],
                image_info['name'],
                channel
            )
            filename = get_fullname(
                image_info['date'], image_info['name'],
                channel, image_info['ix'], image_info['iy']
            )
            image_path = get_filepath(
                data_path,
                dataset,
                'images',
                filename,
                file_type='tiff'
            )

            image_tensor = read_tensor(image_path)
            if image_tensor.ndim == 2:
                image_tensor = image_tensor.reshape(*image_tensor.shape, 1)

            image_tensor = image_tensor / image_tensor.max() * 255
            image_tensors.append(image_tensor.astype(np.uint8))

        image = transforms.ToTensor()(np.concatenate(image_tensors, axis=2))

        prediction = model.predict(image.view(1, count_channels(channels), size, size))

        result = prediction.view(size, size).detach().numpy()

        cv.imwrite(get_filepath(
            predictions_path,
            get_fullname(
                image_info['date'], image_info['name'], 'rgb',
                image_info['ix'], image_info['iy']
            ), file_type='png'
        ), result * 255)


if __name__ == '__main__':
    args = parse_args()
    predict(
        args.data_path, args.model_weights_path,
        args.network, args.test_df, args.save_path,
        args.size, args.channels
    )
