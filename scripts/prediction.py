import argparse
import os

import cv2 as cv
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

from models.utils import get_model


def parse_args():
    parser = argparse.ArgumentParser(
        description='Script for making predictions on test images of dataset.')
    parser.add_argument('--network', '-n', default='unet50')
    parser.add_argument('--datasets_path', '-dp', required=True, help='Path to directory with datasets')
    parser.add_argument('--model_weights_path', '-mwp', required=True, help='Path to file with model weights')
    parser.add_argument('--test_df', '-td', required=True, help='Path to test dataframe')
    parser.add_argument('--save_path', '-sp', required=True, help='Path to save predictions')

    return parser.parse_args()


def predict(datasets_path, model_weights_path, network, test_df_path, save_path):
    model = get_model(network)
    checkpoint = torch.load(model_weights_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])

    test_df = pd.read_csv(test_df_path)

    predictions_path = os.path.join(save_path, "predictions")

    if not os.path.exists(predictions_path):
        os.mkdir(predictions_path)
        print("Prediction directory created.")

    for ind, filename in tqdm(test_df.iterrows()):
        img_path = os.path.join(datasets_path, filename["dataset_folder"], "images", filename['image_name'] + ".tiff")
        img = Image.open(img_path)

        img_tensor = transforms.ToTensor()(img)

        prediction = model.predict(img_tensor.view(1, 3, 224, 224))

        result = prediction.view(224, 224).detach().numpy()

        cv.imwrite(os.path.join(predictions_path, filename['image_name'] + ".png"), result * 255)


if __name__ == '__main__':
    args = parse_args()
    predict(args.datasets_path, args.model_weights_path, args.network, args.test_df, args.save_path)
