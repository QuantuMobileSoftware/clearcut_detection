import os

import cv2 as cv
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

from models.utils import get_model
from params import args


def predict(datasets_path, model_weights_path, network, test_df_path, save_path, channels_number=3):
    model = get_model(network)
    checkpoint = torch.load(model_weights_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])

    test_df = pd.read_csv(test_df_path)

    predictions_path = os.path.join(save_path, "predictions")

    if not os.path.exists(predictions_path):
        os.makedirs(predictions_path, exist_ok=True)
        print("Prediction directory created.")

    for ind, image_info in tqdm(test_df.iterrows()):
        img_path = os.path.join(datasets_path, image_info["dataset_folder"], "images",
                                image_info["name"] + '_' + image_info["channel"] + '_' + image_info["position"] + '.' +
                                image_info["image_type"])
        img = Image.open(img_path)

        img_tensor = transforms.ToTensor()(img)

        prediction = model.predict(
            img_tensor.view(1, channels_number, image_info["image_size"], image_info["image_size"]))

        result = prediction.view(image_info["image_size"], image_info["image_size"]).detach().numpy()

        cv.imwrite(os.path.join(predictions_path, image_info["name"] + '_' + image_info["channel"] + '_' + image_info[
            "position"] + '.png'), result * 255)


def image_predict(model, unlabeled_data, image_name, img_size, channels_number=3):
    image_path = os.path.join(unlabeled_data, image_name)

    img = Image.open(image_path)

    img_tensor = transforms.ToTensor()(img)

    prediction = model.predict(img_tensor.view(1, channels_number, img_size, img_size))

    result = prediction.view(img_size, img_size).detach().numpy()
    return result


if __name__ == '__main__':
    predict(args.datasets_path, args.weights_path, args.network, args.test_df, args.save_path)
