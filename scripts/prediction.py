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
    parser = argparse.ArgumentParser(description='Script for evaluating performance of the model.')
    arg = parser.add_argument

    arg('--datasets_path', '-dsp', help='Path to directory with datasets')
    arg('--weights_path', '-wp', help='Path to file with model weights')
    arg('--test_df', '-tdf', help='Path to test dataframe')
    arg('--save_path', '-sp', help='Path to save predictions')
    arg('--img_size', '-is', type=int, default=224)
    arg('--network', '-n', default='fpn50')
    arg('--channels', '-ch', default=['rgb'], type=list, help='Channels list')

    return parser.parse_args()


def count_channels(channels):
    count = 0
    for ch in channels:
        if ch == 'rgb':
            count += 3
        elif ch == 'ndvi_color':
            count += 4
        elif ch in ['ndvi', 'b2', 'b3', 'b4', 'b8']:
            count += 1
        else:
            raise Exception('{} channel is unknown!'.format(ch))

    return count


def predict(datasets_path, model_weights_path, network, test_df_path, save_path, img_size, channels,
            images_folder="images", image_type="tiff"):
    model = get_model(network)
    checkpoint = torch.load(model_weights_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])

    test_df = pd.read_csv(test_df_path)

    predictions_path = os.path.join(save_path, "predictions")
    if not os.path.exists(predictions_path):
        os.makedirs(predictions_path, exist_ok=True)
        print("Prediction directory created.")

    for ind, image_info in tqdm(test_df.iterrows()):
        img_path = os.path.join(datasets_path, image_info["dataset_folder"], images_folder,
                                image_info["name"] + '_' + image_info["position"] + '.' + image_type)
        img = Image.open(img_path)

        img_tensor = transforms.ToTensor()(img)

        prediction = model.predict(img_tensor.view(1, count_channels(channels), img_size, img_size))

        result = prediction.squeeze().detach().numpy()

        cv.imwrite(os.path.join(predictions_path, image_info["name"] + '_' + image_info["position"] + '.png'),
                   result * 255)


def image_predict(model, unlabeled_data, image_name, img_size, channels_number=3):
    image_path = os.path.join(unlabeled_data, image_name)

    img = Image.open(image_path)

    img_tensor = transforms.ToTensor()(img)

    prediction = model.predict(img_tensor.view(1, channels_number, img_size, img_size).cuda())

    result = prediction.view(img_size, img_size).detach().cpu().numpy()
    return result


if __name__ == '__main__':
    args = parse_args()
    predict(args.datasets_path, args.weights_path, args.network, args.test_df, args.save_path, args.img_size,
            channels=args.channels)
