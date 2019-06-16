import argparse
import os

import cv2 as cv
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

from models.utils import get_model
from params import args


def predict(datasets_path, model_weights_path, network, test_df_path, save_path):
    model = get_model(network)
    checkpoint = torch.load(model_weights_path, map_location='cpu')
    model.load_state_dict(checkpoint)

    test_df = pd.read_csv(test_df_path)

    predictions_path = os.path.join(save_path, "predictions")

    if not os.path.exists(predictions_path):
        os.mkdir(predictions_path)
        print("Prediction directory created.")

    for ind, image_info in tqdm(test_df.iterrows()):
        img_path = os.path.join(datasets_path, image_info["dataset_folder"], "images",
                                image_info["name"] + '_' + image_info["channel"] + '_' + image_info["position"] + '.' +
                                image_info["image_type"])
        img = Image.open(img_path)

        img_tensor = transforms.ToTensor()(img)

        _, prediction = model.predict(img_tensor.view(1, 3, image_info["image_size"], image_info["image_size"]))

        result = prediction.view(image_info["image_size"], image_info["image_size"]).detach().numpy()

        cv.imwrite(os.path.join(predictions_path, image_info["name"] + '_' + image_info["channel"] + '_' + image_info[
            "position"] + '.png'), result * 255)


if __name__ == '__main__':
    predict(args.datasets_path, args.model_weights_path, args.network, args.test_df, args.save_path)
