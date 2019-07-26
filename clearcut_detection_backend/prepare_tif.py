import os
import argparse
import rasterio
import numpy as np

from tqdm import tqdm
from os.path import join, splitext


def search_band(band, files, file_type):
    for file in files:
        if band in file and file.endswith(file_type):
            return splitext(file)[0]
    
    return None


def to_tiff(img_file):
    os.system(
        f'gdal_translate -ot Float32 \
        {img_file} {splitext(img_file)[0]}.tif'
    )


def scale_img(img_file):
    with rasterio.open(img_file) as src:
        img = src.read(1)
        min_ = img.min()
        max_ = min(img.max(), img.mean() + 2 * img.std())
        os.system(
            f'gdal_translate -ot Byte \
            -scale {min_} {max_} 0 255 \
            {img_file} {splitext(img_file)[0]}_scaled.tif'
        )


def get_ndvi(b4_file, b8_file, ndvi_file):
    os.system(
        f'gdal_calc.py -A {b4_file} -B {b8_file} \
        --outfile={ndvi_file} \
        --calc="(B-A)/(A+B+0.1)" --type=Float32'
    )


def parse_args():
    parser = argparse.ArgumentParser(description='Script for predicting masks.')
    parser.add_argument(
        '--peps_folder', '-f', dest='peps_folder',
        required=True, help='Path to downloaded images'
    )
    parser.add_argument(
        '--save_path', '-s', dest='save_path', default='data',
        help='Path to directory where results will be stored'
    )
    return parser.parse_args()


if __name__ == '__main__':
    
    args = parse_args()

    granule_folder = join(args.peps_folder, 'GRANULE')
    tile_folder = list(os.walk(granule_folder))[0][1][-1]
    img_folder = join(granule_folder, tile_folder, 'IMG_DATA', 'R10m')
    img_names = list(os.walk(img_folder))[0][2]
    save_file = join(args.save_path, f'{tile_folder}.tif')

    b2_name = join(img_folder, search_band('B02', img_names, 'jp2'))
    b3_name = join(img_folder, search_band('B03', img_names, 'jp2'))
    b4_name = join(img_folder, search_band('B04', img_names, 'jp2'))
    b8_name = join(img_folder, search_band('B08', img_names, 'jp2'))
    ndvi_name = join(img_folder, 'ndvi')

    print('b4 and b8 bands are converting to *tif...')

    to_tiff(f'{b4_name}.jp2')
    to_tiff(f'{b8_name}.jp2')

    print('ndvi band is processing...')    

    get_ndvi(f'{b4_name}.tif', f'{b8_name}.tif', f'{ndvi_name}.tif')

    print('all bands are scaling to 8-bit images...')

    bands = [
        f'{b4_name}.jp2', f'{b3_name}.jp2', f'{b2_name}.jp2',
        f'{ndvi_name}.tif', f'{b8_name}.jp2'
    ]
    for band in tqdm(bands):
        scale_img(band)

    print('all bands are being merged...')

    os.system(
        f'gdal_merge.py -separate -o {save_file} \
        {b4_name}_scaled.tif {b3_name}_scaled.tif \
        {b2_name}_scaled.tif {ndvi_name}_scaled.tif {b8_name}_scaled.tif'
    )

    for item in os.listdir(img_folder):
        if item.endswith('.tif'):
            os.remove(join(img_folder, item))

    print('temp files have been deleted')