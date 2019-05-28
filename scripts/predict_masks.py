import os
import torch
import argparse
import cv2 as cv
import pandas as pd
import torchvision.transforms as transforms

from PIL import Image
from tqdm import tqdm
from catalyst.dl.utils import UtilsFactory

from pytorch.models.utils import get_model


def get_filename(info_record):
    return '{}_{}_{}'.format(
        info_record['name'],
        info_record['channel'],
        info_record['position'])


def get_full_filename(info_record, type_col='mask_type'):
    return '{}.{}'.format(
        get_filename(info_record),
        info_record[type_col])


def get_image(info_record):
    image_path = os.path.join(
        info_record['image_path'],
        get_full_filename(info_record, 'image_type'))
    
    mask_path = os.path.join(
        info_record['mask_path'],
        get_full_filename(info_record))

    img = Image.open(image_path)
    mask = Image.open(mask_path)

    img_tensor = transforms.ToTensor()(img)
    mask_tensor = transforms.ToTensor()(mask)

    return {'features': img_tensor, 'targets': mask_tensor}


def load_model(network, model_weights_path):
    model = get_model(network)
    checkpoint = torch.load(model_weights_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model.eval()


def predict(model, image_tensor, input_shape=(1, 3, 224, 224)):
    prediction = model(image_tensor.view(*input_shape))
    return torch \
        .sigmoid(prediction.view(*input_shape[2:])) \
        .detach().numpy()


def get_predictions_masks(network, model_weights_path, data_info_path):
    model = load_model(network, model_weights_path)

    data_info = pd.read_csv(data_info_path)

    predictions = {}
    masks = {}
    for _, info_record in tqdm(data_info.iterrows()):
        data = get_image(info_record)
        filename = get_filename(info_record)

        predictions[filename] = predict(model, data['features'])
        masks[filename] = data['targets']
        
    return predictions, masks


def save_predictions(predictions, save_path):
    predictions_path = os.path.join(save_path, 'predictions')
    
    if not os.path.exists(predictions_path):
        os.mkdir(predictions_path)
        print('Pediction directory created.')
        
    for filename, image in predictions.items():
        cv.imwrite(
            os.path.join(
                predictions_path,
                filename + '.png'),
            image * 255)
    
    print('Predictions saved.')


def parse_args():
    parser = argparse.ArgumentParser(description='Script for predicting masks.')
    parser.add_argument('--image_path', '-ip', dest='image_path', required=True,
                        help='Path to source image')
    parser.add_argument('--save_path', '-sp', dest='save_path', default='data',
                        help='Path to directory where pieces will be stored')
    parser.add_argument('--width', '-w', dest='width', default=224, type=int,
                        help='Width of a piece')
    parser.add_argument('--height', '-hgt', dest='height', default=224, type=int,
                        help='Height of a piece')

    return parser.parse_args()


if __name__ == '__main__':
    test_df_path = '../data/test_df.csv'
    dataset_path = '../data'
    network = 'linknet'
    model_weights_path = '../data/best.pth'

    predictions, masks = get_predictions_masks(network, model_weights_path, test_df_path)
    save_predictions(predictions, dataset_path)