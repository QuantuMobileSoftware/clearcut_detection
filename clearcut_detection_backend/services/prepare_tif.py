"""
Conversion raw satellite images to prepared model images
"""
# import argparse
# import os
# from os.path import splitext
import logging
import os
import imageio
import rasterio
from distutils.util import strtobool
import numpy as np
from osgeo import gdal
from tqdm import tqdm
from django.conf import settings
from services.gdal_calc import Calc
from services.gdal_merge import merge_img

prepare_tif = strtobool(os.environ.get('PREPARE_TIF', 'true'))

logger = logging.getLogger('prepare_tif')


# def search_band(band, folder, file_type):
#     for file in os.listdir(folder):
#         if band in file and file.endswith(file_type):
#             return splitext(file)[0]
#
#     return None


# def parse_args():
#     parser = argparse.ArgumentParser(description='Script for predicting masks.')
#     parser.add_argument(
#         '--save_path', '-s', dest='save_path', default='data',
#         help='Path to directory where results will be stored'
#     )
#     return parser.parse_args()


def scale_img(img_file, output_file=None, min_value=0, max_value=255, output_type='Byte'):
    with rasterio.open(img_file) as src:  # TODO FIXIT not optimal, do it by gdal lib
        img = src.read(1)
        img = np.nan_to_num(img)
        mean_ = img.mean()
        std_ = img.std()
        min_ = max(img.min(), mean_ - 2 * std_)
        max_ = min(img.max(), mean_ + 2 * std_)

    # output_file = os.path.splitext(img_file)[0] if output_file is None else output_file
    # logger.info(f'output_file: {output_file}')
    # os.system(
    #     f'gdal_translate -q -ot {output_type} \
    #     -scale {min_} {max_} {min_value} {max_value} \
    #     {img_file} {output_file}_scaled.tif'
    # )
    ds = gdal.Open(str(img_file))
    ds = gdal.Translate(
        f'{output_file}_scaled.tif',
        ds,
        # format='MEM',
        scaleParams=[[min_, max_, min_value, max_value]],
        outputType=gdal.GDT_Byte,
    )
    if ds is None:
        logger.error(f'converting {img_file} to {output_file}_scaled.tif error')
        return 'fail'
    del ds
    return 'success'


# def get_ndvi(b4_file, b8_file, ndvi_file):
#     os.system(
#         f'gdal_calc.py -A {b4_file} -B {b8_file} \
#         --outfile={ndvi_file} \
#         --calc="(B-A)/(A+B+0.001)" --type=Float32 --quiet'
#     )


def create_output_tiffs(save_path, tile):
    """
    defining temporary files names
    """
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


# def to_tiff(input_jp2_file, output_tiff_file, output_type='Float32'):
#     os.system(
#         f'gdal_translate -q -ot {output_type} \
#         {input_jp2_file} {output_tiff_file}'
#     )


def all_bands_to_tif(tile, output_tiffs):
    to_tiff(tile.source_b04_location, output_tiffs.get('tiff_b4_name'))
    to_tiff(tile.source_b08_location, output_tiffs.get('tiff_b8_name'))
    to_tiff(tile.source_b8a_location, output_tiffs.get('tiff_b8a_name'))
    to_tiff(tile.source_b11_location, output_tiffs.get('tiff_b11_name'))
    to_tiff(tile.source_b12_location, output_tiffs.get('tiff_b12_name'))
    to_tiff(tile.source_tci_location, output_tiffs.get('tiff_rgb_name'), gdal.GDT_Byte)


def to_tiff(input_jp2_file, output_tiff_file, output_type=gdal.GDT_Float32):
    ds = gdal.Open(str(input_jp2_file))
    ds = gdal.Translate(str(output_tiff_file), ds, outputType=output_type)
    if ds is None:
        logger.error(f'converting {input_jp2_file} to {output_tiff_file} error')
        return 'fail'
    del ds
    return 'success'


def get_ndvi(input1, input2, outfile):
    calc = "(B-A)/(A+B+0.001)"
    Calc(calc,
         str(outfile),
         A=str(input1),
         B=str(input2),
         NoDataValue=None,
         type='Float32',
         format=None,
         creation_options=None,
         allBands='',
         overwrite=True,
         debug=False,
         quiet=True,
         )
    return


def scale_all(tile, output_tiffs):
    scale_img(output_tiffs.get('tiff_ndvi_name'), output_tiffs.get('scaled_ndvi_name'))
    scale_img(output_tiffs.get('tiff_ndmi_name'), output_tiffs.get('scaled_ndmi_name'))
    scale_img(tile.source_b08_location, output_tiffs.get('scaled_b8_name'))
    scale_img(tile.source_b8a_location, output_tiffs.get('scaled_b8a_name'))
    scale_img(tile.source_b11_location, output_tiffs.get('scaled_b11_name'))
    scale_img(tile.source_b12_location, output_tiffs.get('scaled_b12_name'))
    return


def create_tiff_path(tile):
    """
    create path for tiff images
    """
    save_path = settings.MODEL_TIFFS_DIR / str(tile.tile_index)
    save_path.mkdir(parents=True, exist_ok=True)
    output_folder = save_path / tile.tile_name
    output_folder.mkdir(parents=True, exist_ok=True)
    tiff_output_name = output_folder / f'{tile.tile_name}.tif'
    return save_path, output_folder, tiff_output_name


def prepare_tiff(tile):
    if not prepare_tif:
        convert_to_tiff = 0
        create_ndvi = 0
        scaling = 0
        merge = 0
        create_clouds = 0
        save_in_png = 0
    else:
        convert_to_tiff = 1
        create_ndvi = 1
        scaling = 1
        merge = 1
        create_clouds = 1
        save_in_png = 0

    save_path, output_folder, tiff_output_name = create_tiff_path(tile)  # create path for tiff images

    output_tiffs = create_output_tiffs(save_path, tile)  # defining temporary files names

    if convert_to_tiff:
        logger.info(f'converting all bands to *tif for {tile.tile_name} started')
        try:
            all_bands_to_tif(tile, output_tiffs)
        except (IOError, ValueError, Exception):
            logger.error(f'Error when converting all bands to *tif for {tile.tile_name}\n\n', exc_info=True)
            return save_path, tile.tile_name, None
        logger.info(f'converting all bands to *tif for {tile.tile_name} finished')

    if create_ndvi:
        logger.info(f'creating ndvi band for {tile.tile_name} started')
        try:
            get_ndvi(output_tiffs.get('tiff_b4_name'), output_tiffs.get('tiff_b8_name'), output_tiffs.get('tiff_ndvi_name'))
        except (IOError, ValueError, Exception):
            logger.error(f'Error when creating ndvi band for {tile.tile_name}\n\n', exc_info=True)
            return save_path, tile.tile_name, None
        logger.info(f'creating ndvi band for {tile.tile_name} finished')

        logger.info(f'creating ndmi band for {tile.tile_name} started')
        try:
            get_ndvi(output_tiffs.get('tiff_b11_name'), output_tiffs.get('tiff_b8a_name'), output_tiffs.get('tiff_ndmi_name'))
        except (IOError, ValueError, Exception):
            logger.error(f'Error when creating ndmi band for {tile.tile_name}\n\n', exc_info=True)
            return save_path, tile.tile_name, None
        logger.info(f'creating ndmi band for {tile.tile_name} finished')

    if scaling:
        logger.info(f'scaling all bands to 8-bit images for {tile.tile_name} started')
        try:
            scale_all(tile, output_tiffs)
        except (IOError, ValueError, Exception):
            logger.error(f'Error when scaling all bands to 8-bit images for {tile.tile_name}\n\n', exc_info=True)
            return save_path, tile.tile_name, None
        logger.info(f'scaling all bands to 8-bit images for {tile.tile_name} finished')

    if merge:
        logger.info(f'merge all bands for {tile.tile_name} started')
        # os.system(
        #     f"gdal_merge.py -separate -o {tiff_output_name} \
        #     {output_tiffs.get('tiff_rgb_name')} \
        #     {output_tiffs.get('scaled_b8_name')}_scaled.tif {output_tiffs.get('scaled_b8a_name')}_scaled.tif \
        #     {output_tiffs.get('scaled_b11_name')}_scaled.tif {output_tiffs.get('scaled_b12_name')}_scaled.tif \
        #     {output_tiffs.get('scaled_ndvi_name')}_scaled.tif {output_tiffs.get('scaled_ndmi_name')}_scaled.tif"
        # )
        try:
            merge_img(f"{tiff_output_name}",
                      f"{output_tiffs.get('tiff_rgb_name')}",
                      f"{output_tiffs.get('scaled_b8_name')}_scaled.tif",
                      f"{output_tiffs.get('scaled_b8a_name')}_scaled.tif",
                      f"{output_tiffs.get('scaled_b11_name')}_scaled.tif",
                      f"{output_tiffs.get('scaled_b12_name')}_scaled.tif",
                      f"{output_tiffs.get('scaled_ndvi_name')}_scaled.tif",
                      f"{output_tiffs.get('scaled_ndmi_name')}_scaled.tif",
                      is_verbose=0,
                      is_quiet=0,
                      separate=1,
                      frmt='GTiff'
                      )
            logger.info(f'merge all bands for {tile.tile_name} finished')
        except (IOError, ValueError, Exception):
            logger.error(f'Error when merge all bands for {tile.tile_name} \n\n', exc_info=True)
            return save_path, tile.tile_name, None

    if create_clouds:
        logger.info(f'creating clouds.tiff for {tile.tile_name} started')
        try:
            to_tiff(tile.source_clouds_location, output_folder / 'clouds.tiff')
        except (IOError, ValueError, Exception):
            logger.error(f'Error when creating clouds.tiff for {tile.tile_name} \n\n', exc_info=True)
        logger.info(f'creating clouds.tiff for {tile.tile_name} finished')
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
    tile.is_prepared = 1
    tile.save()

    return save_path, tile.tile_name, str(tile.tile_index)
