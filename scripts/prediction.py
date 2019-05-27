import os

import cv2 as cv
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from catalyst.dl.utils import UtilsFactory
from tqdm import tqdm

import argparse

from models.utils import get_model


def parse_args():
    parser = argparse.ArgumentParser(
        description='Script for evaluating performance of the model.')
    parser.add_argument(
        '--network', '-n', dest='network', default='linknet')
    parser.add_argument(
        '--dataset_path', '-dp', dest='dataset_path', required=True, help='Path to dataset directory')
    parser.add_argument(
        '--model_weights_path', '-mwp', dest='model_weights_path', required=True, help='Path to file with model weights')
    parser.add_argument(
        '--test_df', '-td', dest='test_df', required=True, help='Path to test dataframe')
    return parser.parse_args()


def predict(dataset_path, model_weights_path, network, test_df_path):

    model = get_model(network)
    checkpoint = torch.load(model_weights_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    test_df = pd.read_csv(test_df_path)

    images_path = os.path.join(dataset_path, "images")
    predictions_path = os.path.join(dataset_path, "predictions")

    if not os.path.exists(predictions_path):
        os.mkdir(predictions_path)
        print("Prediction directory created.")

    for ind, filename in tqdm(test_df.iterrows()):
        img_path = os.path.join(images_path, filename['image_name'] + ".tiff")
        img = Image.open(img_path)

        img_tensor = transforms.ToTensor()(img)

        prediction = model(img_tensor.view(1, 3, 224, 224))

        result = torch.sigmoid(prediction.view(224, 224)).detach().numpy()

        cv.imwrite(os.path.join(predictions_path, filename['image_name'] + ".png"), result * 255)


if __name__ == '__main__':
    args = parse_args()
    predict(args.dataset_path, args.model_weights_path, args.network, args.test_df)
