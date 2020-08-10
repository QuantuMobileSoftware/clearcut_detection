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
from services.model_call import ModelCaller
from services.sentinel_download import SentinelDownload
from services.upload_to_mapbox import start_upload
from services.configuration import area_tile_set

sentinel_download = strtobool(os.environ.get('SENTINEL_DOWNLOAD', 'true'))
convert_to_tiff = strtobool(os.environ.get('CONVERT_TO_TIFF', 'true'))
mapbox_upload = strtobool(os.environ.get('UPLOAD_TO_MAPBOX', 'true'))
call_model = 1


logger = logging.getLogger('update')

if __name__ == '__main__':
    # Tile.objects.exclude(tile_index__in=area_tile_set).update(is_tracked=0)
    for tile_index in area_tile_set:
        tile, created = Tile.objects.get_or_create(tile_index=tile_index)  # TODO
        tile.is_tracked = 1
        tile.save()

    if sentinel_download:
        sentinel_downloader = SentinelDownload()
        sentinel_downloader.process_download()
    logger.info('Sentinel pictures were downloaded')

    if call_model:
        model_caller = ModelCaller()
        model_caller.start()
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
