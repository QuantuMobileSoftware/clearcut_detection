"""
Updating mapbox tiles
"""
import os
import logging
from distutils.util import strtobool
import django
django.setup()
from clearcuts.models import Tile
from services.jp2_to_tiff_conversion import jp2_to_tiff
from tiff_prepare.services import ModelCaller
from services.upload_to_mapbox import start_upload
from services.configuration import area_tile_set
from downloader.services import SentinelDownload

sentinel_download = strtobool(os.environ.get('SENTINEL_DOWNLOAD', 'true'))
convert_to_tiff = strtobool(os.environ.get('CONVERT_TO_TIFF', 'true'))
mapbox_upload = strtobool(os.environ.get('UPLOAD_TO_MAPBOX', 'true'))
call_model = 1


logger = logging.getLogger('update')

if __name__ == '__main__':
    # Tile.objects.exclude(tile_index__in=area_tile_set).update(is_tracked=0)
    for tile in Tile.objects.filter(is_tracked=1).order_by('tile_index'):

        if sentinel_download:
            sentinel_downloader = SentinelDownload(tile.tile_index)
            sentinel_downloader.process_download()
        logger.info(f'Sentinel pictures for {tile.tile_index} were downloaded')

        if call_model:
            model_caller = ModelCaller(tile.tile_index)
            model_caller.start()
            exit(0)
        if convert_to_tiff:
            logger.info('Start convert jp2_to_tiff')
            jp2_to_tiff()
            logger.info('Convert jp2_to_tiff finished')

        if mapbox_upload:
            try:
                logger.info('Start uploading to mapbox')
                uploader = start_upload()
            except (IOError, ValueError, FileNotFoundError, FileExistsError, Exception):
                logger.error('Error\n\n', exc_info=True)
