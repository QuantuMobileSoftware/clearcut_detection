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
from tqdm import tqdm


def load_model(network, model_weights_path, channels):
    model = get_model(network)
    model.encoder.conv1 = torch.nn.Conv2d(
        count_channels(channels), 64, kernel_size=(7, 7),
        stride=(2, 2), padding=(3, 3), bias=False
    )
    checkpoint = torch.load(model_weights_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model, device = UtilsFactory.prepare_model(model.eval())

    return model, device


def predict(model, image_tensor, input_shape, device):
    prediction = model(image_tensor.view(input_shape)).to(device)
    return torch \
        .sigmoid(prediction.view(input_shape[2:])) \
        .cpu().detach().numpy()


def predict_raster(tiff_file, channels, network, model_weights_path, input_size=224):
    model, device = load_model(network, model_weights_path, channels)
    with rasterio.open(tiff_file) as src:
        meta = src.meta
        meta['count'] = 1
        raster_array = np.zeros((src.meta['height'], src.meta['width']))
        xs = src.bounds.left
        window_size_meters = input_size * (src.res[0])
        window_size_pixels = input_size
        cnt = 0
        pbar = tqdm()
        while xs < src.bounds.right:
            ys = src.bounds.bottom
            while ys < src.bounds.top:
                row, col = src.index(xs, ys)
                pbar.set_postfix(Row=f'{row}', Col=f'{col}')
                step_row = row - int(window_size_pixels)
                step_col = col + int(window_size_pixels)

                if step_row < 0:
                    row = int(window_size_pixels)
                    step_row = 0

                right_bound = src.index(src.bounds.right, 0)[1]
                if step_col > right_bound:
                    col = int(right_bound - window_size_pixels)
                    step_col = right_bound

                res = np.moveaxis(src.read(window=((step_row, row), (col, step_col))), 0, -1)
                res = filter_by_channels(res, channels)
                rect = [
                    [step_row, row],
                    [col, step_col]
                ]

                #for channel in range(res.shape[0]):
                #    res[channel] = scale(res[channel], 255)

                res = transforms.ToTensor()(res.astype(np.uint8)).to(device)
                pred = predict(model, res, (1, count_channels(channels), input_size, input_size), device)
                stack_arr = np.dstack([
                    pred[rect[0][0] - rect[0][1]:, :rect[1][1] - rect[1][0]],
                    raster_array[rect[0][0]:rect[0][1], rect[1][0]:rect[1][1]]])
                raster_array[rect[0][0]:rect[0][1], rect[1][0]:rect[1][1]] = np.mean(stack_arr, axis=2)

                pbar.update(1)
                cnt += 1
                ys = ys + 0.5 * window_size_meters

            xs = xs + 0.5 * window_size_meters

        src.close()

    meta['dtype'] = 'float32'

    return raster_array.astype(np.float32), meta


def get_model(name='fpn50', model_weights_path=None):
    if 'unet50' in name:
        return smp.Unet('resnet50', encoder_weights='imagenet')
    elif 'unet101' in name:
        return smp.Unet('resnet101', encoder_weights='imagenet')
    elif 'fpn50' in name:
        return smp.FPN('resnet50', encoder_weights='imagenet')
    elif 'fpn101' in name:
        return smp.FPN('resnet101', encoder_weights='imagenet')
    else:
        raise ValueError("Unknown network")


def count_channels(channels):
    count = 0
    for ch in channels:
        if ch == 'rgb' or ch == 'nrg':
            count += 3
        elif ch in ['ndvi', 'b8']:
            count += 1
        else:
            raise Exception('{} channel is unknown!'.format(ch))

    return count


def filter_by_channels(image_tensor, channels):
    result = []
    for ch in channels:
        if ch == 'rgb' or ch == 'nrg':
            result.append(image_tensor[:, :, :3])
        elif ch == 'ndvi':
            result.append(image_tensor[:, :, 3:4])
        elif ch == 'b8':
            result.append(image_tensor[:, :, 4:5])
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
            meta.update(driver='GTiff')
            meta.update(dtype=rasterio.float32)
            dst.write(raster_array, i)


def polygonize(raster_array, meta, threshold=0.7, transform=True):
    raster_array = raster_array > threshold
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
    gc.to_file(os.path.join(save_path, f'{filename}.geojson'), driver='GeoJSON')


def parse_args():
    parser = argparse.ArgumentParser(description='Script for predicting masks.')
    parser.add_argument(
        '--image_path', '-ip', dest='image_path',
        required=True, help='Path to source image'
    )
    parser.add_argument(
        '--model_weights_path', '-mwp', dest='model_weights_path',
        required=True, help='Path to directory where pieces will be stored'
    )
    parser.add_argument(
        '--network', '-net', dest='network', default='unet50',
        help='Model architecture'
    )
    parser.add_argument(
        '--save_path', '-sp', dest='save_path', default='../data/predicted',
        help='Path to directory where results will be stored'
    )
    parser.add_argument(
        '--channels', '-ch', dest='channels',
        default=['nrg'],
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

    filename = re.split(r'[./]', args.image_path)[-2]
    predicted_filename = f'predicted_{filename}'

    if not args.polygonize_only:
        raster_array, meta = predict_raster(
            args.image_path, args.channels,
            args.network, args.model_weights_path
        )
        save_raster(raster_array, meta, args.save_path, filename)
    else:
        with rasterio.open(os.path.join(args.save_path, f'{predicted_filename}.tif')) as src:
            raster_array = src.read()
            raster_array = np.moveaxis(raster_array, 0, -1)
            meta = src.meta
            src.close()

    polygons = polygonize(raster_array, meta, args.threshold)
    save_polygons(polygons, meta, args.save_path, predicted_filename)


if __name__ == '__main__':
    main()
