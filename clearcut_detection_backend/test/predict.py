import os
import re
import cv2
import torch
import logging
import imageio
import rasterio
import argparse
import geopandas
import numpy as np
import segmentation_models_pytorch as smp

from catalyst.dl.utils import UtilsFactory
from geopandas import GeoSeries
from scipy import spatial
from shapely.geometry import Polygon
from shapely.ops import unary_union
from torchvision import transforms
from torch import nn
from tqdm import tqdm
from rasterio.windows import Window
from rasterio.plot import reshape_as_image
from rasterio import features
from skimage.transform import match_histograms
from scipy.ndimage import gaussian_filter

from test_data_prepare import get_gt_polygons
from settings import MODEL_TIFFS_DIR, DATA_DIR
from utils import path_exists_or_create, area_tile_set_test

CLOUDS_PROBABILITY_THRESHOLD = 15
NEAREST_POLYGONS_NUMBER = 10
DATES_FOR_TILE = 2

import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(format='%(asctime)s %(message)s')

def load_model(network, model_weights_path, channels, neighbours):
    device = 'gpu' if torch.cuda.is_available() else 'cpu'
    model = get_model(network)
    model.encoder.conv1 = nn.Conv2d(
        count_channels(channels)*neighbours, 64, kernel_size=(7, 7),
        stride=(2, 2), padding=(3, 3), bias=False
    )
    model, device = UtilsFactory.prepare_model(model)
    model.load_state_dict(torch.load(model_weights_path, map_location=torch.device(device)))
    return model, device


def predict(image_tensor, model, channels, neighbours, size, device):
    image_shape = 1, count_channels(channels)*neighbours, size, size
    prediction, _ = model.predict(image_tensor.view(image_shape).to(device, dtype=torch.float))
    result = prediction.view(size, size).detach().cpu().numpy()
    return result


def diff(img1, img2):
    img2 = match_histograms(img2, img1, multichannel=True)
    difference = ( (img1 - img2) / (img1 + img2) )
    difference = (difference + 1) * 127
    return np.concatenate((difference.astype(np.uint8), img1.astype(np.uint8), img2.astype(np.uint8)), axis=-1)


def mask_postprocess(mask):
    kernel = np.ones((3, 3), np.uint8)
    erosion = cv2.erode(mask, kernel, iterations = 1)
    kernel = np.ones((5, 5), np.uint8)
    closing = cv2.morphologyEx(erosion, cv2.MORPH_CLOSE, kernel)
    return closing


def predict_raster(img_path, channels, network, model_weights_path, save_path, input_size=56, neighbours=3):
    tile = os.path.basename(img_path)
    tiff_files = [os.path.join(img_path, f'{tile}_{i}', f'{tile}_{i}_merged.tiff') for i in range(DATES_FOR_TILE)]
    model, device = load_model(network, model_weights_path, channels, neighbours)

    with rasterio.open(tiff_files[0]) as source_current, \
         rasterio.open(tiff_files[1]) as source_previous:

        meta = source_current.meta
        meta['count'] = 1
        clearcut_mask = np.zeros((source_current.height, source_current.width))
        image = np.zeros((source_current.height, source_current.width))
        mask = np.ones((source_current.height, source_current.width))

        gt_polygons_filename = get_gt_polygons()
        gt_polygons = geopandas.read_file(gt_polygons_filename)
        gt_polygons = gt_polygons.to_crs(source_current.crs)
        mask = features.rasterize(shapes=gt_polygons['geometry'],
                                out_shape=(source_current.height, source_current.width),
                                transform=source_current.transform,
                                default_value=1)
        
        for i in tqdm(range(source_current.width // input_size)):
            for j in range(source_current.height // input_size):                
                bottom_row = j * input_size
                upper_row = (j + 1) * input_size
                left_column = i * input_size
                right_column = (i + 1) * input_size

                if mask[left_column:right_column, bottom_row:upper_row].sum() > 0:
                    corners=[
                        source_current.xy(bottom_row, left_column),
                        source_current.xy(bottom_row, right_column),
                        source_current.xy(upper_row, right_column),
                        source_current.xy(upper_row, left_column),
                        source_current.xy(bottom_row, left_column)
                        ]

                    window = Window(bottom_row, left_column, input_size, input_size)
                    image_current = reshape_as_image(source_current.read(window=window))
                    image_previous = reshape_as_image(source_previous.read(window=window))

                    difference_image = diff(image_current, image_previous)
                    image_tensor = transforms.ToTensor()(difference_image.astype(np.uint8)).to(device, dtype=torch.float)

                    predicted = predict(image_tensor, model, channels, neighbours, input_size, device)
                    predicted = mask_postprocess(predicted)
                    clearcut_mask[left_column:right_column, bottom_row:upper_row] += predicted

                    mask_piece = mask[left_column:right_column, bottom_row:upper_row]
                    # mask_piece = (gaussian_filter(mask_piece, 0.5) > 0)
                    
                    cv2.imwrite(f"{save_path}/preds/{i}_{j}.png", predicted * 255)
                    cv2.imwrite(f"{save_path}/masks/{i}_{j}.png", mask_piece * 255)




    meta['dtype'] = 'float32'
    return clearcut_mask.astype(np.float32), meta


def get_model(name, classification_head=True, model_weights_path=None):
    if name == 'unet_ch':
        aux_params = dict(
            pooling='max',             # one of 'avg', 'max'
            dropout=0.1,               # dropout ratio, default is None
            activation='sigmoid',      # activation function, default is None
            classes=1,                 # define number of output labels
        )
        return smp.Unet('resnet18', aux_params=aux_params, 
                        encoder_weights=None, encoder_depth=2, 
                        decoder_channels=(256, 128))    
    else:
        raise ValueError("Unknown network")


def count_channels(channels):
    count = 0
    for ch in channels:
        if ch == 'rgb':
            count += 3
        elif ch in ['ndvi', 'ndmi', 'b8', 'b8a', 'b11', 'b12']:
            count += 1
        else:
            raise Exception('{} channel is unknown!'.format(ch))

    return count


def filter_by_channels(image_tensor, channels, neighbours=3):
    # order: ['TCI','B08','B8A','B11','B12', 'NDVI', 'NDMI']
    result = []
    for i in range(neighbours):
        for ch in channels:
            if ch == 'rgb':
                result.append(image_tensor[:, :, (0+i*9):(3+i*9)])
            elif ch == 'b8':
                result.append(image_tensor[:, :, (3+i*9):(4+i*9)])
            elif ch == 'b8a':
                result.append(image_tensor[:, :, (4+i*9):(5+i*9)])
            elif ch == 'b11':
                result.append(image_tensor[:, :, (5+i*9):(6+i*9)])
            elif ch == 'b12':
                result.append(image_tensor[:, :, (6+i*9):(7+i*9)])
            elif ch == 'ndvi':
                result.append(image_tensor[:, :, (7+i*9):(8+i*9)])
            elif ch == 'ndmi':
                result.append(image_tensor[:, :, (8+i*9):(9+i*9)])
            else:
                raise Exception(f'{ch} channel is unknown!')
    return np.concatenate(result, axis=2)


def scale(tensor, max_value):
    max_ = tensor.max()
    if max_ > 0:
        return tensor / max_ * max_value
    return tensor


def save_raster(raster_array, meta, save_path, filename):
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
        logging.info("Data directory created.")

    save_path = os.path.join(save_path, f'predicted_{filename}')

    cv2.imwrite(f'{save_path}.png', raster_array)

    with rasterio.open(f'{save_path}.tif', 'w', **meta) as dst:
        for i in range(1, meta['count'] + 1):
            dst.write(raster_array, i)


def polygonize(raster_array, meta, transform=True, mode=cv2.RETR_TREE):
    raster_array = (raster_array * 255).astype(np.uint8)

    contours, _ = cv2.findContours(raster_array, mode, cv2.CHAIN_APPROX_SIMPLE)

    polygons = []
    for i in tqdm(range(len(contours))):
        c = contours[i]
        n_s = (c.shape[0], c.shape[2])
        if n_s[0] > 2:
            if transform:
                polys = [tuple(i) * meta['transform'] for i in c.reshape(n_s)]
            else:
                polys = [tuple(i) for i in c.reshape(n_s)]
            polygons.append(Polygon(polys))

    return polygons


def save_polygons(polygons, save_path, filename):
    if len(polygons) == 0:
        logging.info('no_polygons detected')
        return

    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
        logging.info("Data directory created.")

    logging.info(f'{filename}.geojson saved.')
    polygons.to_file(os.path.join(save_path, f'{filename}.geojson'), driver='GeoJSON')


def intersection_poly(test_poly, mask_poly):
    intersecion_score = False
    if test_poly.is_valid and mask_poly.is_valid:
        intersection_result = test_poly.intersection(mask_poly)
        if not intersection_result.is_valid:
            intersection_result = intersection_result.buffer(0)
        if not intersection_result.is_empty:
            intersecion_score = True
    return intersecion_score


def polygon_to_geodataframe(polygons, src_crs):
    polygons = {'geometry': polygons}
    return geopandas.GeoDataFrame(polygons, crs=src_crs)


def parse_args():
    parser = argparse.ArgumentParser(description='Script for predicting masks.')
    parser.add_argument(
        '--image_path', '-ip', dest='image_path',
        default=f'{MODEL_TIFFS_DIR}/{area_tile_set_test.pop()}', help='Path to source image'
    )
    parser.add_argument(
        '--model_weights_path', '-mwp', dest='model_weights_path',
        default=f'{DATA_DIR}/unet_v4.pth', help='Path to directory where pieces will be stored'
    )
    parser.add_argument(
        '--network', '-net', dest='network', default='unet_ch',
        help='Model architecture'
    )
    parser.add_argument(
        '--save_path', '-sp', dest='save_path', default=f'{DATA_DIR}/predicted',
        help='Path to directory where results will be stored'
    )
    parser.add_argument(
        '--channels', '-ch', dest='channels',
        default=['rgb', 'b8', 'b8a', 'b11', 'b12', 'ndvi', 'ndmi'],
        help='Channel list', nargs='+'
    )
    parser.add_argument(
        '--threshold', '-t', dest='threshold',
        default=0.4, help='Threshold to get binary values in mask', type=float
    )
    parser.add_argument(
        '--polygonize_only', '-po', dest='polygonize_only',
        default=False, help='Flag to skip prediction', type=bool
    )

    return parser.parse_args()


def main():
    args = parse_args()
    path_exists_or_create(args.save_path)
    path_exists_or_create(f"{args.save_path}/preds")
    path_exists_or_create(f"{args.save_path}/masks")
    filename = re.split(r'[./]', args.image_path)[-1]
    predicted_filename = f'predicted_{filename}'

    raster_array, meta = predict_raster(
        args.image_path,
        args.channels, args.network, args.model_weights_path,
        args.save_path
    )
    # save_raster(raster_array, meta, args.save_path, filename)

    clearcuts = polygonize(raster_array > args.threshold, meta)
    clearcuts = polygon_to_geodataframe(clearcuts, meta['crs'])
    save_polygons(clearcuts, args.save_path, predicted_filename)


if __name__ == '__main__':
    main()
