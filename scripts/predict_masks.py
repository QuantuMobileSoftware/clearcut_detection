import os
from collections import defaultdict

import numpy as np
import imageio
import rasterio
import torch
import argparse
import cv2 as cv
import pandas as pd
import cv2

from PIL import Image
from tqdm import tqdm
from pytorch.utils import get_model
from torchvision.transforms import ToTensor
from shapely.geometry import Polygon, MultiPolygon

from preprocessing.image_division import divide_into_pieces


def read_tensor(filepath):
    return imageio.imread(filepath)


def get_fullname(*name_parts):
    return '_'.join(tuple(map(str, name_parts)))


def join_pathes(*pathes):
    return os.path.join(*pathes)


def get_filepath(*path_parts, file_type):
    return '{}.{}'.format(join_pathes(*path_parts), file_type)


def count_channels(channels):
    count = 0
    for ch in channels:
        if ch == 'rgb':
            count += 3
        elif ch == 'ndvi':
            count += 1
        elif ch == 'ndvi_color':
            count += 4
        elif ch == 'b2':
            count += 1
        else:
            raise Exception('{} channel is unknown!'.format(ch))

    return count


def get_image(image_path, filename):
    image_path = os.path.join(image_path, filename)
    img = Image.open(image_path)

    return ToTensor()(img)


def get_input_pair(
    data_info_row, channels, data_path,
    image_folder, mask_folder,
    image_type, mask_type
):
    if len(channels) == 0:
        raise Exception('You have to specify at least one channel.')

    image_tensors = []
    for channel in channels:
        dataset = get_fullname(
            data_info_row['date'],
            data_info_row['name'],
            channel
        )
        filename = get_fullname(
            data_info_row['date'], data_info_row['name'],
            channel, data_info_row['ix'], data_info_row['iy']
        )
        image_path = get_filepath(
            data_path,
            dataset,
            image_folder,
            filename,
            file_type=image_type
        )

        image_tensor = read_tensor(image_path)
        if image_tensor.ndim == 2:
            image_tensor = image_tensor.reshape(*image_tensor.shape, 1)

        image_tensor = image_tensor / image_tensor.max() * 255
        image_tensors.append(image_tensor.astype(np.uint8))

    mask_path = get_filepath(
        data_path,
        dataset,
        mask_folder,
        filename,
        file_type=mask_type
    )

    masks = ToTensor()(read_tensor(mask_path))
    images = ToTensor()(np.concatenate(image_tensors, axis=2))

    return {'features': images, 'targets': masks}


def load_model(network, model_weights_path, channels):
    model = get_model(network)
    model.encoder.conv1 = torch.nn.Conv2d(
        count_channels(channels), 64, kernel_size=(7, 7),
        stride=(2, 2), padding=(3, 3), bias=False
    )
    checkpoint = torch.load(model_weights_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model.eval()


def get_filenames(path):
    return tuple(os.walk(path))[0][2]


def predict(model, image_tensor, input_shape=(1, 3, 224, 224)):
    prediction = model(image_tensor.view(*input_shape))
    return torch \
        .sigmoid(prediction.view(*input_shape[2:])) \
        .detach().numpy()


def get_predictions(network, model_weights_path, images_path, channels):
    model = load_model(network, model_weights_path, channels)
    predictions = []
    for filename in tqdm(get_filenames(images_path)):
        predictions.append(
            predict(model, get_image(images_path, filename)))

    return predictions


def save_predictions(predictions, filenames, save_path):
    predictions_path = os.path.join(save_path, 'predictions')

    if not os.path.exists(predictions_path):
        os.mkdir(predictions_path)
        print('Pediction directory created.')

    for pred, filename in zip(predictions, filenames):
        cv.imwrite(
            os.path.join(
                predictions_path,
                filename + '.png'),
            pred * 255)

    print('Predictions saved.')
    return predictions_path


def concat_predictions(predictions_path, pieces_info_path):
    pieces_info = pd.read_csv(pieces_info_path, index_col=0)

    result = []
    row = []
    for piece in tqdm(pieces_info.iterrows()):
        if piece[1]['start_x'] == 0 and piece[1]['start_y'] != 0:
            result.append(np.concatenate(row, axis=1))
            row = []
        filename = piece[1]['piece_image'][:-5] + '.png'
        with Image.open(os.path.join(predictions_path, filename)) as img:
            pred = ToTensor()(img).numpy()[0]
        row.append(pred)

    result.append(np.concatenate(row, axis=1))

    return np.concatenate(result, axis=0)


def save_image(save_path, mask):
    imageio.imwrite(save_path, mask)


def save_raster(raster_array, meta, save_path):
    with rasterio.open(save_path, 'w', **meta) as dst:
        for i in range(1, meta['count'] + 1):
            src_array = raster_array[i - 1]
            dst.write(src_array, i)


def mask_to_polygons(mask, epsilon=10., min_area=10.):
    # first, find contours with cv2: it's much faster than shapely
    image, contours, hierarchy = cv2.findContours(
        ((mask == 1) * 255).astype(np.uint8),
        cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
    # create approximate contours to have reasonable submission size
    approx_contours = [cv2.approxPolyDP(cnt, epsilon, True)
                       for cnt in contours]
    if not contours:
        return MultiPolygon()
    # now messy stuff to associate parent and child contours
    cnt_children = defaultdict(list)
    child_contours = set()
    assert hierarchy.shape[0] == 1
    # http://docs.opencv.org/3.1.0/d9/d8b/tutorial_py_contours_hierarchy.html
    for idx, (_, _, _, parent_idx) in enumerate(hierarchy[0]):
        if parent_idx != -1:
            child_contours.add(idx)
            cnt_children[parent_idx].append(approx_contours[idx])
    # create actual polygons filtering by area (removes artifacts)
    all_polygons = []
    for idx, cnt in enumerate(approx_contours):
        if idx not in child_contours and cv2.contourArea(cnt) >= min_area:
            assert cnt.shape[1] == 1
            poly = Polygon(
                shell=cnt[:, 0, :],
                holes=[c[:, 0, :] for c in cnt_children.get(idx, [])
                       if cv2.contourArea(c) >= min_area])
            all_polygons.append(poly)
    # approximating polygons might have created invalid ones, fix them
    all_polygons = MultiPolygon(all_polygons)
    if not all_polygons.is_valid:
        all_polygons = all_polygons.buffer(0)
        # Sometimes buffer() converts a simple Multipolygon to just a Polygon,
        # need to keep it a Multi throughout
        if all_polygons.type == 'Polygon':
            all_polygons = MultiPolygon([all_polygons])

    return all_polygons


def parse_args():
    parser = argparse.ArgumentParser(description='Script for predicting masks.')
    parser.add_argument(
        '--image_path', '-ip', dest='image_path',
        required=True, help='Path to source image'
    )
    parser.add_argument(
        '--save_path', '-sp', dest='save_path', default='data',
        help='Path to directory where pieces will be stored'
    )
    parser.add_argument(
        '--width', '-w', dest='width', default=224,
        type=int, help='Width of a piece'
    )
    parser.add_argument(
        '--height', '-hgt', dest='height', default=224,
        type=int, help='Height of a piece'
    )
    parser.add_argument(
        '--data_path', '-dp', dest='data_path',
        default='../data', help='Path to the data'
    )
    parser.add_argument(
        '--save_path', '-sp', dest='save_path',
        default='../data',
        help='Path to directory where data will be stored'
    )
    parser.add_argument(
        '--images_folder', '-imf', dest='images_folder',
        default='images',
        help='Name of folder where images are storing'
    )
    parser.add_argument(
        '--masks_folder', '-mf', dest='masks_folder',
        default='masks',
        help='Name of folder where masks are storing'
    )
    parser.add_argument(
        '--instances_folder', '-inf', dest='instances_folder',
        default='instance_masks',
        help='Name of folder where instances are storing'
    )
    parser.add_argument(
        '--image_type', '-imt', dest='image_type',
        default='tiff',
        help='Type of image file'
    )
    parser.add_argument(
        '--mask_type', '-mt', dest='mask_type',
        default='png',
        help='Type of mask file'
    )
    parser.add_argument(
        '--instance_type', '-int', dest='instance_type',
        default='geojson',
        help='Type of instance file'
    )
    parser.add_argument(
        '--channels', '-ch', dest='channels',
        default=['rgb', 'ndvi', 'ndvi_color', 'b2'],
        help='Channel list', type=list
    )

    return parser.parse_args()


if __name__ == '__main__':
    tif_path = '../data/20160103_66979721-be1b-4451-84e0-4a573236defd_rgb.tif'
    pieces_path = '../data/pieces'
    w, h = 224, 224
    with rasterio.open(tif_path, 'r') as src:
        meta = src.meta

    # divide_into_pieces(tif_path, pieces_path, w, h)
    images_path = os.path.join(pieces_path, 'images')
    pieces_names = tuple(os.walk(images_path))[0][2]
    pieces_names = list(map(lambda x: x[:-5], pieces_names))

    test_df_path = '../data/test_df.csv'
    dataset_path = '../data'
    network = 'unet50'
    model_weights_path = '../logs/unet50_rgb/checkpoints/best.pth'
    channels = ['rgb']

    # predictions = get_predictions(
    #     network, model_weights_path,
    #     images_path, channels)
    # predictions_path = save_predictions(predictions, pieces_names, dataset_path)
    predictions_path = os.path.join(dataset_path, 'predictions')
    predicted_mask_path = os.path.join(dataset_path, 'predicted_mask.png')
    # raster_array = concat_predictions(
    #     predictions_path,
    #     os.path.join(pieces_path, 'image_pieces.csv')
    # )
    # save_image(
    #     predicted_mask_path,
    #     raster_array
    # )

    threshold = 0.3
    raster_array = imageio.imread(predicted_mask_path)
    raster_array = raster_array > threshold
    raster_array = raster_array.astype(np.uint8)

    meta['width'] = raster_array.shape[1]
    meta['height'] = raster_array.shape[0]
    meta['count'] = 1
    raster_path = os.path.join(dataset_path, 'raster.tiff')
    save_raster(raster_array, meta, raster_path)



