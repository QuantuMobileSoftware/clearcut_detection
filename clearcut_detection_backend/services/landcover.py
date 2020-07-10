import logging
from shutil import copyfile
from utils import download_without_progress, fetch_file_from_zip, fetch_all_from_zip
from requests.exceptions import (HTTPError, InvalidURL, ConnectionError)
from zipfile import (BadZipFile, LargeZipFile)
from rasterio import open as r_open, band

from rasterio.warp import calculate_default_transform, reproject, Resampling
from django.conf import settings

logger = logging.getLogger('landcover')


class Landcover:
    tif = 'E020N60_ProbaV_LC100_epoch2015_global_v2.0.2_forest-type-layer_EPSG-4326.tif'

    def __init__(self):
        self.data_path = settings.DATA_DIR
        self.landcover_path = settings.LAND_TIFF_DIR
        self.landcover_path.mkdir(parents=True, exist_ok=True)
        self.forest_tiff = self.landcover_path / 'forest.tiff'

    def create_forest_corr(self, tiff_output_name):
        src = r_open(tiff_output_name)
        lnd = r_open(self.forest_tiff)
        if src.crs != lnd.crs:
            self.transform_crs(self.forest_tiff, self.landcover_path / 'forest_corr.tiff', dst_crs=src.crs)
        src.close()
        lnd.close()

    @staticmethod
    def transform_crs(data_path, save_path, dst_crs='epsg:32636'):
        with r_open(data_path) as src:
            transform, width, height = calculate_default_transform(src.crs,
                                                                   dst_crs,
                                                                   src.width,
                                                                   src.height,
                                                                   *src.bounds)
            kwargs = src.meta.copy()
            kwargs.update({'crs': dst_crs,
                           'transform': transform,
                           'width': width,
                           'height': height})
            with r_open(save_path, 'w', **kwargs) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=band(src, i),
                        destination=band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=dst_crs,
                        resampling=Resampling.nearest)
            dst.close()
        src.close()

    @staticmethod
    def download_landcover(url):
        """
        download file from url
        :param url: str
        :return: BytesIO
        """
        file = None
        try:
            logger.info(f'start download zip file from {url}')
            file = download_without_progress(url)
        except (HTTPError, InvalidURL, ConnectionError, ConnectionError):
            logger.error(f'cant download zip file from {url}\n\n', exc_info=True)
            exit(1)
        return file

    @staticmethod
    def unzip_landcover(file, landcover_path):
        """
        extract all files from archive
        :param file: BytesIO or file path
        :param landcover_path: file path
        :return:
        """
        try:
            fetch_all_from_zip(file, landcover_path)
        except (BadZipFile, LargeZipFile):
            logger.error(f'cant unzip files to {landcover_path}\n\n', exc_info=True)
            exit(1)
        return

    @staticmethod
    def extract_file(file, source, destination):
        """
        extract specific file from archive
        :param file: BytesIO or file path
        :param source: file name to be extracted
        :param destination: path for extraction
        :return:
        """
        try:
            fetch_file_from_zip(file, source, destination)
        except (BadZipFile, LargeZipFile, Exception):
            logger.error(f'cant unzip {source} to {destination}\n\n', exc_info=True)
            exit(1)

    @staticmethod
    def copy_file(source, destination):
        """
        copy file from source to destination
        :param source: file path
        :param destination: file path
        :return:
        """
        try:
            copyfile(source, destination)
        except (OSError, Exception):
            logger.error(f'cant copy {source} to {destination}\n\n', exc_info=True)
            exit(1)
