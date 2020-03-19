"""
Conversion raw satellite images to prepared model images
"""
import argparse
import os
from os.path import join, splitext

import imageio
import rasterio
import numpy as np

from tqdm import tqdm

from clearcuts.models import TileInformation
from sentinel_download import DOWNLOADED_IMAGES_DIR
from utils import path_exists_or_create


def search_band(band, folder, file_type):
    for file in os.listdir(folder):
        if band in file and file.endswith(file_type):
            return splitext(file)[0]

    return None


def to_tiff(input_jp2_file, output_tiff_file, output_type='Float32'):
    os.system(
        f'gdal_translate -ot {output_type} \
        {input_jp2_file} {output_tiff_file}'
    )


def scale_img(img_file, output_file=None, min_value=0, max_value=255, output_type='Byte'):
    with rasterio.open(img_file) as src:
        img = src.read(1)
        img = np.nan_to_num(img)
        mean_ = img.mean()
        std_ = img.std()
        min_ = max(img.min(), mean_ - 2 * std_)
        max_ = min(img.max(), mean_ + 2 * std_)

        output_file = os.path.splitext(img_file)[0] if output_file is None else output_file

        os.system(
            f'gdal_translate -ot {output_type} \
            -scale {min_} {max_} {min_value} {max_value} \
            {img_file} {output_file}_scaled.tif'
        )


def get_ndvi(b4_file, b8_file, ndvi_file):
    os.system(
        f'gdal_calc.py -A {b4_file} -B {b8_file} \
        --outfile={ndvi_file} \
        --calc="(B-A)/(A+B+0.001)" --type=Float32'
    )


def parse_args():
    parser = argparse.ArgumentParser(description='Script for predicting masks.')
    parser.add_argument(
        '--data_folder', '-f', dest='data_folder',
        required=True, help='Path to downloaded images'
    )
    parser.add_argument(
        '--save_path', '-s', dest='save_path', default='data',
        help='Path to directory where results will be stored'
    )
    return parser.parse_args()


MODEL_TIFFS_DIR = path_exists_or_create('data/model_tiffs')


def prepare_tiff(data_folder=DOWNLOADED_IMAGES_DIR, save_path=MODEL_TIFFS_DIR):
    for tile_name in TileInformation.objects.values_list('tile_name', flat=True):

        source_files = {'b4_name': join(data_folder, f'{tile_name}_B04.jp2'),
                        'b8_name': join(data_folder, f'{tile_name}_B08.jp2'),
                        'rgb_name': join(data_folder, f'{tile_name}_TCI.jp2')}

        output_tiffs = {'tiff_b4_name': join(save_path, f'{tile_name}_B04.tif'),
                        'tiff_b8_name': join(save_path, f'{tile_name}_B08.tif'),
                        'tiff_rgb_name': join(save_path, f'{tile_name}_TCI.tif'),
                        'tiff_ndvi_name': join(save_path, f'{tile_name}_ndvi.tif'),
                        'tiff_output_name': join(save_path, f'{tile_name}.tif'),
                        'scaled_b8_name': join(save_path, f'{tile_name}_B08'),
                        'scaled_ndvi_name': join(save_path, f'{tile_name}_ndvi')}

        print('\nb4 and b8 bands are converting to *tif...\n')

        to_tiff(source_files.get('b4_name'), output_tiffs.get('tiff_b4_name'))
        to_tiff(source_files.get('b8_name'), output_tiffs.get('tiff_b8_name'))
        to_tiff(source_files.get('rgb_name'), output_tiffs.get('tiff_rgb_name'), 'Byte')

        print('\nndvi band is processing...')

        get_ndvi(output_tiffs.get('tiff_b4_name'),
                 output_tiffs.get('tiff_b8_name'),
                 output_tiffs.get('tiff_ndvi_name'))

        print('\nall bands are scaling to 8-bit images...\n')

        scale_img(output_tiffs.get('tiff_ndvi_name'), output_tiffs.get('scaled_ndvi_name'))
        scale_img(source_files.get('b8_name'), output_tiffs.get('scaled_b8_name'))

        print('\nall bands are being merged...\n')

        os.system(
            f"gdal_merge.py -separate -o {output_tiffs.get('tiff_output_name')} \
            {output_tiffs.get('tiff_rgb_name')} {output_tiffs.get('scaled_ndvi_name')}_scaled.tif "
            f"{output_tiffs.get('scaled_b8_name')}_scaled.tif"
        )

        print('\nsaving in png...\n')

        png_folder = path_exists_or_create(os.path.join(MODEL_TIFFS_DIR, f'{tile_name}_png'))

        bands = {
            f'{join(png_folder, "rgb.png")}': output_tiffs.get('tiff_rgb_name'),
            f'{join(png_folder, "ndvi.png")}': output_tiffs.get('tiff_ndvi_name'),
            f'{join(png_folder, "b8.png")}': f"{output_tiffs.get('scaled_b8_name')}_scaled.tif"
        }

        for dest, source in tqdm(bands.items()):
            with rasterio.open(source) as src:
                imageio.imwrite(dest, np.moveaxis(src.read(), 0, -1))
                src.close()

        for item in os.listdir(save_path):
            if item.endswith('.tif'):
                os.remove(join(save_path, item))

        print('\ntemp files have been deleted\n')


if __name__ == '__main__':
    args = parse_args()

    prepare_tiff(args.data_folder, args.save_path)
