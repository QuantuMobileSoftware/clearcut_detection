"""
Conversion raw satellite images to prepared model images
"""
import logging
import rasterio
import numpy as np
from osgeo import gdal
from tiff_prepare.gdal_calc import Calc
from tiff_prepare.gdal_merge import merge_img


logger = logging.getLogger('prepare_tif')


def scale_img(img_file, output_file=None, min_value=0, max_value=255, output_type='Byte', prepared=None):
    try:
        with rasterio.open(img_file) as src:  # TODO FIXIT not optimal, do it by gdal lib
            img = src.read(1)
            img = np.nan_to_num(img)
            mean_ = img.mean()
            std_ = img.std()
            min_ = max(img.min(), mean_ - 2 * std_)
            max_ = min(img.max(), mean_ + 2 * std_)

        ds = gdal.Open(str(img_file))
        ds = gdal.Translate(
            f'{output_file}_scaled.tif',
            ds,
            # format='MEM',
            scaleParams=[[min_, max_, min_value, max_value]],
            outputType=gdal.GDT_Byte,
        )
    except (IOError, ValueError, Exception):
        logger.error(
            f'Error when scaling {output_file} for {prepared.tile.tile_index}-{prepared.image_date}\n\n', exc_info=True
        )
        prepared.success = -1
        prepared.save()
        return 'fail'
    if ds is not None:
        del ds
        return 'success'

    else:
        logger.error(f'converting {img_file} to {output_file}_scaled.tif error')
        prepared.success = -1
        prepared.save()
        return 'fail'


def to_tiff(input_jp2_file, output_tiff_file, output_type=gdal.GDT_Float32, prepared=None):
    logger.info(f'started converting to {output_tiff_file} with output_type={output_type}')
    ds = gdal.Open(str(input_jp2_file))
    ds = gdal.Translate(str(output_tiff_file), ds, outputType=output_type)
    if ds is not None:
        del ds
        logger.info(f'converting to {output_tiff_file} with output_type={output_type} finished')
        if str(output_tiff_file).endswith('clouds.tif'):
            prepared.cloud_tiff_location = output_tiff_file
            prepared.save()
        return 'success'
    else:
        logger.error(f'converting {input_jp2_file} to {output_tiff_file} error')
        prepared.success = -1
        prepared.save()
        return 'fail'


def get_ndvi(input1, input2, outfile, prepared=None):
    calc = "(B-A)/(A+B+0.001)"
    try:
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
    except (IOError, ValueError, Exception):
        logger.error(
            f'Error when creating ndvi band for {prepared.tile.tile_index}-{prepared.image_date}\n\n', exc_info=True
        )
        prepared.success = -1
        prepared.save()
    return


def merge_img_extra(*args, **kwargs):
    prepared = kwargs.pop('prepared')
    tile_by_date = kwargs.pop('tile_by_date')
    try:
        merge_img(*args)
    except (IOError, ValueError, Exception):
        logger.error(
            f'Error when merging bands for {args[0]}\n\n', exc_info=True
        )
        prepared.success = -1
        prepared.save()
        return

    prepared.success = 1
    prepared.model_tiff_location = args[0]
    prepared.save()
    tile_by_date.is_new = 0
    tile_by_date.save()
    return args[0]
