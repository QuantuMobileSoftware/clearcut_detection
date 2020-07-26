import os
import re
import cv2
import torch
import logging
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
from skimage.transform import match_histograms

from utils import LandcoverPolygons

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


def predict(model, image_tensor, input_shape, device):
    preds, _ = model(image_tensor.view(input_shape))
    prediction = preds.to(device)
    return torch \
        .sigmoid(prediction.view(input_shape[2:])) \
        .cpu().detach().numpy()


def diff(img1, img2):
    img2 = match_histograms(img2, img1, multichannel=True)
    difference = ( (img1 - img2) / (img1 + img2) )
    difference = (difference + 1) * 127
    return np.concatenate((difference.astype(np.uint8), img1.astype(np.uint8), img2.astype(np.uint8)), axis=-1)


def window_from_extent(corners, aff):
    xmax=max(corners[0][0],corners[2][0])
    xmin=min(corners[0][0],corners[2][0])
    ymax=max(corners[0][1],corners[2][1])
    ymin=min(corners[0][1],corners[2][1])
    col_start, row_start = ~aff * (xmin, ymax)
    col_stop,  row_stop  = ~aff * (xmax, ymin)
    return ((int(row_start), int(row_stop)), (int(col_start), int(col_stop)))

def predict_raster(img_path, channels, network, model_weights_path, input_size=56, neighbours=3):
    tile = os.path.basename(img_path)
    tiff_files = [os.path.join(img_path, f'{tile}_{i}', f'{tile}_{i}.tif') for i in range(DATES_FOR_TILE)]
    model, device = load_model(network, model_weights_path, channels, neighbours)

    with rasterio.open(tiff_files[0]) as source_current, \
         rasterio.open(tiff_files[1]) as source_previous:

        meta = source_current.meta
        meta['count'] = 1
        clearcut_mask = np.zeros((source_current.height, source_current.width))
        # pbar = tqdm()
        # for i in range(source_current.width // input_size):
        #     for j in range(source_current.height // input_size):
        #         pbar.set_postfix(row=f'{i}', col=f'{j}', num_pixels=f'{clearcut_mask.sum()}')
        #
        #         bottom_row = j * input_size
        #         upper_row = (j + 1) * input_size
        #         left_column = i * input_size
        #         right_column = (i + 1) * input_size
        #
        #         corners=[
        #             source_current.xy(bottom_row, left_column),
        #             source_current.xy(bottom_row, right_column),
        #             source_current.xy(upper_row, right_column),
        #             source_current.xy(upper_row, left_column),
        #             source_current.xy(bottom_row, left_column)
        #             ]
        #
        #         window = Window(bottom_row, left_column, input_size, input_size)
        #         image_current = np.moveaxis(source_current.read(window=window), 0, -1)
        #         image_previous = np.moveaxis(source_previous.read(window=window), 0, -1)
        #
        #         merged_image = diff(image_current, image_previous)
        #         filtered_image = filter_by_channels(merged_image, channels)
        #
        #         image_tensor = transforms.ToTensor()(filtered_image.astype(np.uint8)).to(device)
        #
        #         n_channels = count_channels(channels) * neighbours
        #         image_shape = (1, n_channels, input_size, input_size)
        #         predicted = predict(model, image_tensor, image_shape, device)
        #
        #         clearcut_mask[bottom_row:upper_row, left_column:right_column] = predicted
        #     pbar.update(1)

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

def morphological_transform(img):
    kernel = np.ones((5,5),np.uint8)
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    kernel = np.ones((3,3),np.uint8)
    closing = cv2.dilate(closing,kernel,iterations = 1)
    return closing

def postprocessing(img_path, clearcuts, src_crs):

    def get_intersected_polygons(polygons, masks, mask_column_name):
        """Finding in GeoDataFrame with clearcuts the masked polygons.

        :param polygons: GeoDataFrame with clearcuts and mask columns
        :param masks: list of masks (e.g., polygons of clouds)
        :param mask_column_name: name of mask column in polygons GeoDataFrame

        :return: GeoDataFrame with filled mask flags in corresponding column
        """
        masked_values = []
        if len(masks) > 0:
            centroids = [[mask.centroid.x, mask.centroid.y] for mask in masks]
            kdtree = spatial.KDTree(centroids)
            for _, clearcut in polygons.iterrows():
                polygon = clearcut['geometry']
                _, idxs = kdtree.query(polygon.centroid, k=NEAREST_POLYGONS_NUMBER)
                masked_value = 0
                for idx in idxs:
                    if intersection_poly(polygon, masks[idx].buffer(0)):
                        masked_value = 1
                        break
                masked_values.append(masked_value)
        polygons[mask_column_name] = masked_values
        return polygons

    tile = os.path.basename(img_path)

    landcover = LandcoverPolygons(tile, src_crs)
    forest_polygons = landcover.get_polygon()

    cloud_files = [f"{img_path}/{tile}_{i}/clouds.tiff" for i in range(DATES_FOR_TILE)]
    cloud_polygons = []
    for cloud_file in cloud_files:
        with rasterio.open(cloud_file) as src:
            clouds = src.read(1)
            meta = src.meta
        clouds = morphological_transform(clouds)
        clouds = (clouds > CLOUDS_PROBABILITY_THRESHOLD).astype(np.uint8)
        if clouds.sum() > 0:
            cloud_polygons.extend(polygonize(clouds, meta, mode=cv2.RETR_LIST))
    
    n_clearcuts = len(clearcuts)
    polygons = {'geometry': clearcuts,
                'forest': np.zeros(n_clearcuts),
                'clouds': np.zeros(n_clearcuts)}

    polygons = geopandas.GeoDataFrame(polygons, crs=src_crs)
    
    # TODO: cloud polygons do not filter all cases correctly, need to investigate
    polygons = get_intersected_polygons(polygons, cloud_polygons, 'clouds')
    polygons = get_intersected_polygons(polygons, forest_polygons, 'forest')
    return polygons


def parse_args():
    parser = argparse.ArgumentParser(description='Script for predicting masks.')
    parser.add_argument(
        '--image_path', '-ip', dest='image_path',
        type=str, required=True, help='Path to source image'
    )
    parser.add_argument(
        '--model_weights_path', '-mwp', dest='model_weights_path',
        default='unet_v4.pth', help='Path to directory where pieces will be stored'
    )
    parser.add_argument(
        '--network', '-net', dest='network', default='unet_ch',
        help='Model architecture'
    )
    parser.add_argument(
        '--save_path', '-sp', dest='save_path', default='predicted',
        help='Path to directory where results will be stored'
    )
    parser.add_argument(
        '--channels', '-ch', dest='channels',
        default=['rgb', 'b8', 'b8a', 'b11', 'b12', 'ndvi', 'ndmi'],
        help='Channel list', nargs='+'
    )
    parser.add_argument(
        '--threshold', '-t', dest='threshold',
        default=0.3, help='Threshold to get binary values in mask', type=float
    )
    parser.add_argument(
        '--polygonize_only', '-po', dest='polygonize_only',
        default=False, help='Flag to skip prediction', type=bool
    )

    return parser.parse_args()


def main():
    args = parse_args()

    filename = re.split(r'[./]', args.image_path)[-1]
    predicted_filename = f'predicted_{filename}'

    if not args.polygonize_only:
        raster_array, meta = predict_raster(
            args.image_path,
            args.channels, args.network, args.model_weights_path
        )
        save_raster(raster_array, meta, args.save_path, filename)
    else:
        with rasterio.open(os.path.join(args.save_path, f'{predicted_filename}.tif')) as src:
            raster_array = src.read()
            raster_array = np.moveaxis(raster_array, 0, -1)
            meta = src.meta
            src.close()

    logging.info('Polygonize raster array of clearcuts...')
    clearcuts = polygonize(raster_array > args.threshold, meta)
    logging.info('Filter polygons of clearcuts')
    polygons = postprocessing(args.image_path, clearcuts, meta['crs'])
    
    save_polygons(polygons, args.save_path, predicted_filename)


if __name__ == '__main__':
    main()
