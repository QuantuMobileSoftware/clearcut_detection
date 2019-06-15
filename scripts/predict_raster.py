import os

import cv2
import torch
import imageio
import rasterio
import numpy as np
import torchvision.transforms as transforms

from tqdm import tqdm
from pytorch.utils import get_model


def read_tensor(filepath):
    return imageio.imread(filepath)


def join_name(*name_parts):
    return '_'.join(tuple(map(str, name_parts)))


def join_pathes(*pathes):
    return os.path.join(*pathes)


def get_filepath(*path_parts, file_type):
    return '{}.{}'.format(join_pathes(*path_parts), file_type)


def get_filenames(path):
    return tuple(os.walk(path))[0][2]


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


def load_model(network, model_weights_path, channels):
    model = get_model(network)
    model.encoder.conv1 = torch.nn.Conv2d(
        count_channels(channels), 64, kernel_size=(7, 7),
        stride=(2, 2), padding=(3, 3), bias=False
    )
    checkpoint = torch.load(model_weights_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])

    return model.eval()


def predict(model, image_tensor, input_shape=(1, 3, 224, 224)):
    prediction = model(image_tensor.view(input_shape))
    return torch \
        .sigmoid(prediction.view(input_shape[2:])) \
        .detach().numpy()


def predict_raster(
        tiff_file, channels, network, model_weights_path,
        window_size=2240, input_size=224
):
    model = load_model(network, model_weights_path, channels)
    with rasterio.open(tiff_file) as src:
        meta = src.meta
        meta['count'] = 1
        meta['dtype'] = 'float32'
        raster_array = np.zeros((src.meta['height'], src.meta['width']), np.float32)
        xs = src.bounds.left
        window_size_meters = window_size
        window_size_pixels = window_size / (src.res[0])
        cnt = 0
        pbar = tqdm()
        while xs < src.bounds.right:
            ys = src.bounds.bottom
            while ys < src.bounds.top:
                row, col = src.index(xs, ys)
                pbar.set_postfix(Row=f'{row}', Col=f'{col}')
                step_row = row - int(window_size_pixels)
                step_col = col + int(window_size_pixels)
                res = src.read(
                    window=(
                        (max(0, step_row), row),
                        (col, step_col)
                    )
                )
                rect = [[max(0, step_row), row], [col, step_col]]
                temp = res.copy()
                res = np.zeros((res.shape[0], input_size, input_size))
                res[:, -temp.shape[1]:, :temp.shape[2]] = temp
                res = res.astype(np.uint8)

                for channel in range(res.shape[2]):
                    res[:, :, channel] = scale(res[:, :, channel], 255)

                res = transforms.ToTensor()(res)
                pred = predict(model, res, (1, count_channels(channels), input_size, input_size))
                stack_arr = np.dstack([
                    pred[rect[0][0] - rect[0][1]:, :rect[1][1] - rect[1][0]],
                    raster_array[rect[0][0]:rect[0][1], rect[1][0]:rect[1][1]]])
                raster_array[rect[0][0]:rect[0][1], rect[1][0]:rect[1][1]] = np.amax(stack_arr, axis=2)

                pbar.update(1)
                cnt += 1
                ys = ys + 0.5 * window_size_meters

            xs = xs + 0.5 * window_size_meters

        src.close()

    return raster_array, meta


def scale(tensor, max_value):
    return (tensor / tensor.max() * max_value).astype(np.uint8)


def save_raster(raster_array, meta, save_path, filename, threshold=0.3):
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
        print("Data directory created.")

    raster_array = raster_array > threshold
    raster_array = (raster_array * 255).astype(np.uint8)

    save_path = os.path.join(save_path, f'predicted_{filename}')

    cv2.imwrite(f'{save_path}.png', raster_array)

    with rasterio.open(f'{save_path}.tif', 'w', **meta) as dst:
        for i in range(1, meta['count'] + 1):
            src_array = raster_array[i - 1]
            dst.write(src_array, i)


def main():
    tiff_file = '../data/20160103_66979721-be1b-4451-84e0-4a573236defd.tif'
    network = 'unet50'
    model_weights_path = '../logs/unet50_rgb_ndvi_ndvi_color_b2/checkpoints/best.pth'
    channels = ['rgb', 'ndvi', 'ndvi_color', 'b2']

    raster_array, meta = predict_raster(tiff_file, channels, network, model_weights_path)
    save_raster(
        raster_array, meta, '../data',
        '20160103_66979721-be1b-4451-84e0-4a573236defd',
    )

if __name__ == '__main__':
    main()
