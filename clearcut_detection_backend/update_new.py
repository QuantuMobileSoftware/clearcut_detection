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
from services.jp2_to_tiff_conversion import jp2_to_tiff, Converter
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


def fetch_new_data():
    for tile in Tile.objects.filter(is_tracked=1, first_date__isnull=False).order_by('tile_index'):

        if sentinel_download:
            sentinel_downloader = SentinelDownload(tile.tile_index)
            sentinel_downloader.request_google_cloud_storage_for_new_data()
            sentinel_downloader.launch_download_pool()
            logger.info(f'Sentinel pictures for {tile.tile_index} were downloaded')

        if convert_to_tiff:
            logger.info(f'start convert for {tile.tile_index}')
            converter = Converter(tile.tile_index)
            converter.convert_all_unconverted_to_tif()
            logger.info(f'finish convert for {tile.tile_index}')

        if prepare_tif:
            img_preprocessing = ImgPreprocessing(tile.tile_index)
            img_preprocessing.start()

        if add_tasks:
            update_task = CreateUpdateTask(tile.tile_index)
            prepared = update_task.get_new_prepared
            update_task.run_from_prepared(prepared)

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


if __name__ == '__main__':
    fetch_new_data()
