"""
Conversion raw satellite images to prepared model images
"""
import argparse
import os
from os.path import splitext
import logging

import imageio
import rasterio
import numpy as np

from tqdm import tqdm
from django.conf import settings

save_in_png = settings.SAVE_IN_PNG

logger = logging.getLogger('prepare_tif')


def search_band(band, folder, file_type):
    for file in os.listdir(folder):
        if band in file and file.endswith(file_type):
            return splitext(file)[0]

    return None


def to_tiff(input_jp2_file, output_tiff_file, output_type='Float32'):
    os.system(
        f'gdal_translate -q -ot {output_type} \
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
            f'gdal_translate -q -ot {output_type} \
            -scale {min_} {max_} {min_value} {max_value} \
            {img_file} {output_file}_scaled.tif'
        )


def get_ndvi(b4_file, b8_file, ndvi_file):
    os.system(
        f'gdal_calc.py -A {b4_file} -B {b8_file} \
        --outfile={ndvi_file} \
        --calc="(B-A)/(A+B+0.001)" --type=Float32 --quiet'
    )


def parse_args():
    parser = argparse.ArgumentParser(description='Script for predicting masks.')
    parser.add_argument(
        '--save_path', '-s', dest='save_path', default='data',
        help='Path to directory where results will be stored'
    )
    return parser.parse_args()


def create_output_tiffs(save_path, tile):
    return {'tiff_b4_name': save_path / f'{tile.tile_name}_B04.tif',
            'tiff_b8_name': save_path / f'{tile.tile_name}_B08.tif',
            'tiff_b8a_name': save_path / f'{tile.tile_name}_B8A.tif',
            'tiff_b11_name': save_path / f'{tile.tile_name}_B11.tif',
            'tiff_b12_name': save_path / f'{tile.tile_name}_B12.tif',
            'tiff_rgb_name': save_path / f'{tile.tile_name}_TCI.tif',
            'tiff_ndvi_name': save_path / f'{tile.tile_name}_ndvi.tif',
            'tiff_ndmi_name': save_path / f'{tile.tile_name}_ndmi.tif',
            'tiff_clouds_name': save_path / f'{tile.tile_name}_clouds.tif',

            'scaled_b8_name': save_path / f'{tile.tile_name}_B08.tif',
            'scaled_b8a_name': save_path / f'{tile.tile_name}_B8A.tif',
            'scaled_b11_name': save_path / f'{tile.tile_name}_B11.tif',
            'scaled_b12_name': save_path / f'{tile.tile_name}_B12.tif',
            'scaled_ndvi_name': save_path / f'{tile.tile_name}_ndvi.tif',
            'scaled_ndmi_name': save_path / f'{tile.tile_name}_ndmi.tif',
            }


def create_bands(output_folder, output_tiffs):
    return {
        f'{output_folder / "rgb.png"}': output_tiffs.get('tiff_rgb_name'),
        f'{output_folder / "b8.png"}': f"{output_tiffs.get('scaled_b8_name')}_scaled.tif",
        f'{output_folder / "b8a.png"}': f"{output_tiffs.get('scaled_b8a_name')}_scaled.tif",
        f'{output_folder / "b11.png"}': f"{output_tiffs.get('scaled_b11_name')}_scaled.tif",
        f'{output_folder / "b12.png"}': f"{output_tiffs.get('scaled_b12_name')}_scaled.tif",
        f'{output_folder / "ndvi.png"}': output_tiffs.get('tiff_ndvi_name'),
        f'{output_folder / "ndmi.png"}': output_tiffs.get('tiff_ndmi_name'),
    }


def prepare_tiff(tile):
    save_path = settings.MODEL_TIFFS_DIR / tile.tile_index
    save_path.mkdir(parents=True, exist_ok=True)

    output_folder = save_path / tile.tile_name
    output_folder.mkdir(parents=True, exist_ok=True)

    output_tiffs = create_output_tiffs(save_path, tile)  # defining temporary files names

    logger.info('\nbands are converting to *tif...\n')
    to_tiff(tile.source_b04_location, output_tiffs.get('tiff_b4_name'))
    to_tiff(tile.source_b08_location, output_tiffs.get('tiff_b8_name'))
    to_tiff(tile.source_b8a_location, output_tiffs.get('tiff_b8a_name'))
    to_tiff(tile.source_b11_location, output_tiffs.get('tiff_b11_name'))
    to_tiff(tile.source_b12_location, output_tiffs.get('tiff_b12_name'))
    to_tiff(tile.source_tci_location, output_tiffs.get('tiff_rgb_name'), 'Byte')

    logger.info('\nndvi band is processing...')
    get_ndvi(output_tiffs.get('tiff_b4_name'),
             output_tiffs.get('tiff_b8_name'),
             output_tiffs.get('tiff_ndvi_name'))

    logger.info('\nndmi band is processing...')
    get_ndvi(output_tiffs.get('tiff_b11_name'),
             output_tiffs.get('tiff_b8a_name'),
             output_tiffs.get('tiff_ndmi_name'))

    logger.info('\nall bands are scaling to 8-bit images...\n')
    scale_img(output_tiffs.get('tiff_ndvi_name'), output_tiffs.get('scaled_ndvi_name'))
    scale_img(output_tiffs.get('tiff_ndmi_name'), output_tiffs.get('scaled_ndmi_name'))
    scale_img(tile.source_b08_location, output_tiffs.get('scaled_b8_name'))
    scale_img(tile.source_b8a_location, output_tiffs.get('scaled_b8a_name'))
    scale_img(tile.source_b11_location, output_tiffs.get('scaled_b11_name'))
    scale_img(tile.source_b12_location, output_tiffs.get('scaled_b12_name'))

    tiff_output_name = output_folder / f'{tile.tile_name}.tif'

    logger.info('\nall bands are being merged...\n')
    os.system(
        f"gdal_merge.py -separate -o {tiff_output_name} \
        {output_tiffs.get('tiff_rgb_name')} \
        {output_tiffs.get('scaled_b8_name')}_scaled.tif {output_tiffs.get('scaled_b8a_name')}_scaled.tif \
        {output_tiffs.get('scaled_b11_name')}_scaled.tif {output_tiffs.get('scaled_b12_name')}_scaled.tif \
        {output_tiffs.get('scaled_ndvi_name')}_scaled.tif {output_tiffs.get('scaled_ndmi_name')}_scaled.tif"
    )
    to_tiff(tile.source_clouds_location, output_folder / 'clouds.tiff')
    tile.model_tiff_location = tiff_output_name
    tile.save()

    if save_in_png:
        logger.info('\nsaving in png...\n')
        bands = create_bands(output_folder, output_tiffs)
        for dest, source in tqdm(bands.items()):
            with rasterio.open(source) as src:
                imageio.imwrite(dest, np.moveaxis(src.read(), 0, -1))
                src.close()

    save_path = settings.MODEL_TIFFS_DIR / f"{tile.tile_name.split('_')[0]}"
    save_path.mkdir(parents=True, exist_ok=True)

    output_folder = save_path / tile.tile_name
    output_folder.mkdir(parents=True, exist_ok=True)

    tiff_output_name = output_folder / f'{tile.tile_name}.tif'

    tile.model_tiff_location = tiff_output_name
    tile.save()

    # if not (settings.LAND_TIFF_DIR / 'forest_corr.tiff').exists():
    #     from services.landcover import Landcover
    #     Landcover().create_forest_corr(tiff_output_name)

    return save_path, tile.tile_name, tiff_output_name
