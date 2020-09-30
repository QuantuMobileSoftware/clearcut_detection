"""
Convert all unconverted to tif
"""
import os
import logging
import django
django.setup()
from services.jp2_to_tiff_conversion import Converter
from clearcuts.models import Tile

logger = logging.getLogger('update')


def convert_all_to_tiff():
    for tile in Tile.objects.filter(is_tracked=1, first_date__isnull=False).order_by('tile_index'):
        logger.info(f'start convert for {tile.tile_index}')
        converter = Converter(tile.tile_index)
        converter.convert_all_unconverted_to_tif()
        logger.info(f'finish convert for {tile.tile_index}')


if __name__ == '__main__':
    convert_all_to_tiff()
