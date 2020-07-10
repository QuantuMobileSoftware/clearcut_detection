"""
Updating mapbox tiles
"""
import logging
import django
django.setup()
from django.conf import settings
from services.landcover import Landcover
from services.jp2_to_tiff_conversion import jp2_to_tiff
from services.model_call import ModelCaller
from services.sentinel_download import SentinelDownload
from services.upload_to_mapbox import start_upload

download_landcover = 0
sentinel_download = 0
call_model = 1
convert_to_tiff = 1
mapbox_upload = 1


logger = logging.getLogger('update')

if __name__ == '__main__':
    landcover = Landcover()
    if download_landcover and not landcover.forest_tiff.exists():
        logger.info(f'file {landcover.forest_tiff} not exists')
        raw_file = landcover.download_landcover(settings.LANDCOVER_URL)
        landcover.extract_file(raw_file, landcover.tif, landcover.data_path)
        landcover.copy_file(landcover.data_path / landcover.tif, landcover.forest_tiff)
        # Path.unlink(landcover.data_path / landcover.tif) # TODO make Landcover.remove_temp_files, and Landcover.start
        logger.info(f'file {landcover.forest_tiff} downloaded from {settings.LANDCOVER_URL}')
    else:
        logger.info(f'file {landcover.forest_tiff} already exists,\n skip loading from {settings.LANDCOVER_URL}')

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
