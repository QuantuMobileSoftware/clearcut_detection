"""
Collects all data starting from settings.START_DATE_FOR_SCAN till now
and creates polygons of forest cutting.
"""
import os
import logging
from distutils.util import strtobool
import django
django.setup()
from clearcuts.models import Tile
from clearcuts.services import CreateUpdateTask
from services.configuration import area_tile_set
from services.jp2_to_tiff_conversion import jp2_to_tiff
from tiff_prepare.services import ImgPreprocessing
from services.upload_to_mapbox import start_upload
from downloader.services import SentinelDownload

sentinel_download = strtobool(os.environ.get('SENTINEL_DOWNLOAD', 'true'))
prepare_tif = strtobool(os.environ.get('PREPARE_TIF', 'true'))
add_tasks = strtobool(os.environ.get('ADD_TASKS', 'true'))
convert_to_tiff = strtobool(os.environ.get('CONVERT_TO_TIFF', 'true'))

make_predict = strtobool(os.environ.get('MAKE_PREDICT', 'true'))
mapbox_upload = strtobool(os.environ.get('UPLOAD_TO_MAPBOX', 'true'))

logger = logging.getLogger('update')


if __name__ == '__main__':
    # Tile.objects.exclude(tile_index__in=area_tile_set).update(is_tracked=0)
    for tile_index in area_tile_set:
        tile, created = Tile.objects.get_or_create(tile_index=tile_index)  # TODO
        tile.is_tracked = 1
        tile.first_date = None
        tile.last_date = None
        tile.save()

    for tile in Tile.objects.filter(is_tracked=1, first_date__isnull=True).order_by('tile_index'):

        if sentinel_download:
            sentinel_downloader = SentinelDownload(tile.tile_index)
            sentinel_downloader.process_download()
            logger.info(f'Sentinel pictures for {tile.tile_index} were downloaded')

        if prepare_tif:
            img_preprocessing = ImgPreprocessing(tile.tile_index)
            img_preprocessing.start()

        if add_tasks:
            CreateUpdateTask().run_all_from_prepared(tile.tile_index)
        continue
        # exit(0)

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
