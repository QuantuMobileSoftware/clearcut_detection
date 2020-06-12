import os
import re
import cv2
import torch
import rasterio
import argparse
import numpy as np
import segmentation_models_pytorch as smp

from catalyst.dl.utils import UtilsFactory
from geopandas import GeoSeries
from shapely.geometry import Polygon
from torchvision import transforms
from torch import nn
from tqdm import tqdm
from rasterio.windows import Window
from skimage.transform import match_histograms

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

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


def crop_external_rmap(src, corners):
    aff = src.transform
    window = window_from_extent(corners, aff)
    arr = src.read(1, window=window)
    return arr


def predict_raster(img_path, channels, network, model_weights_path, input_size=56, neighbours=3):
    tile = img_path.split('/')[-1]
    
    tiff_files = [os.path.join(img_path, f'{tile}_{i}', f'{tile}_{i}.tif') for i in range(2)]
    cloud_files = [os.path.join(img_path, f'{tile}_{i}', 'clouds.tiff') for i in range(2)]
    landcover_file = './data/landcover/forest_corr.tiff'

    model, device = load_model(network, model_weights_path, channels, neighbours)
    with rasterio.open(tiff_files[0]) as src, rasterio.open(tiff_files[1]) as src_next, \
         rasterio.open(cloud_files[0]) as cld_1, rasterio.open(cloud_files[1]) as cld_2, \
         rasterio.open(landcover_file) as lnd:
        
        meta = src.meta
        meta['count'] = 1
        raster_array = np.zeros((src.meta['height'], src.meta['width']))
        pbar = tqdm()
        for i in range(src.width // input_size):
            for j in range(src.height // input_size):
                pbar.set_postfix(row=f'{i}', col=f'{j}', num_pixels=f'{raster_array.sum()}')
                corners=[
                    src.xy(j * input_size, i * input_size),
                    src.xy(j * input_size, (i + 1) * input_size),
                    src.xy((j + 1) * input_size, (i + 1) * input_size),
                    src.xy((j + 1) * input_size, i * input_size),
                    src.xy(j * input_size, i * input_size)
                    ]
                
                cld1 = crop_external_rmap(cld_1, corners)
                cld2 = crop_external_rmap(cld_2, corners)
                if (cld1+cld2).max()<20:
                    img1 = np.moveaxis(src.read(window=Window(j * input_size, i * input_size, input_size, input_size)), 0, -1)
                    img2 = np.moveaxis(src_next.read(window=Window(j * input_size, i * input_size, input_size, input_size)), 0, -1)

                    res = diff(img1,img2)
                    res = filter_by_channels(res, channels)

                    res = transforms.ToTensor()(res.astype(np.uint8)).to(device)
                    pred = predict(model, res, (1, count_channels(channels)*neighbours, input_size, input_size), device)

                    land = (crop_external_rmap(lnd, corners) > 0) * 1
                    if land.size-land.sum() > 3:                  
                        raster_array[j * input_size : (j + 1) * input_size, \
                                     i * input_size : (i + 1) * input_size] = pred * (-1)
                    else:
                        raster_array[j * input_size : (j + 1) * input_size, \
                                     i * input_size : (i + 1) * input_size] = pred * 1



                pbar.update(1)

    src.close()
    src_next.close()
    cld_1.close()
    cld_2.close()
    lnd.close()

    meta['dtype'] = 'float32'
    return raster_array.astype(np.float32), meta


def get_model(name, classification_head=True, model_weights_path=None):
    if name == 'unet_ch':
        aux_params=dict(
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
        print("Data directory created.")

    save_path = os.path.join(save_path, f'predicted_{filename}')

    cv2.imwrite(f'{save_path}.png', raster_array)

    with rasterio.open(f'{save_path}.tif', 'w', **meta) as dst:
        for i in range(1, meta['count'] + 1):
            dst.write(raster_array, i)


def polygonize(raster_array, meta, transform=True):
    raster_array = (raster_array * 255).astype(np.uint8)

    contours, _ = cv2.findContours(raster_array, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

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


def save_polygons(polygons, meta, save_path, filename):
    if len(polygons) == 0:
        print('no_polygons detected')
        return

    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
        print("Data directory created.")

    gc = GeoSeries(polygons)
    gc.crs = meta['crs']
    print(f'{filename}.geojson saved.')
    gc.to_file(os.path.join(save_path, f'{filename}.geojson'), driver='GeoJSON')


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

    polygons = polygonize(raster_array > args.threshold, meta)
    polygons_not_forest = polygonize(raster_array < 0, meta)

    save_polygons(polygons, meta, args.save_path, predicted_filename)
    save_polygons(polygons_not_forest, meta, args.save_path, predicted_filename+'_not_forest')


if __name__ == '__main__':
    main()
