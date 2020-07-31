import os
import re
import datetime
import logging
from enum import Enum

from concurrent.futures import ThreadPoolExecutor, as_completed
from google.cloud import storage
from google.cloud.exceptions import NotFound
from xml.dom import minidom
from xml.etree.ElementTree import ParseError

from utils import area_tile_set_test, bands_to_download, date_current_test, date_previous_test
import settings

logger = logging.getLogger('sentinel')


class TillNameError(Exception):
    def __init__(self, till_name):
        self.message = f'{till_name} is not valid till_name'
        Exception.__init__(self, self.message)

    def __str__(self):
        return self.message


class Bands(Enum):
    TCI = 'TCI'
    B04 = 'B04'
    B08 = 'B08'
    B8A = 'B8A'
    B11 = 'B11'
    B12 = 'B12'


class SentinelDownload:

    def __init__(self):
        settings.DOWNLOADED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = './key.json'
        self.sequential_dates_count = settings.MAXIMUM_DATES_STORE_FOR_TILE
        self.area_tile_set = area_tile_set_test
        self.bands_to_download = bands_to_download
        self.base_uri = 'gs://gcp-public-data-sentinel-2'
        self.bucket_name = 'gcp-public-data-sentinel-2'
        self.prefix = 'L2/tiles'
        self.tiles_and_uris_dict = {tile_name: self.get_tile_uri(tile_name) for tile_name in self.area_tile_set}
        self.storage_client = storage.Client()
        self.storage_bucket = self.get_storage_bucket()

        logger.info(f'area tile set for download: {self.area_tile_set}')
        logger.info(f'bands to download:{self.bands_to_download}')
        print(bands_to_download)

    def get_storage_bucket(self):
        try:
            return self.storage_client.get_bucket(self.bucket_name)
        except NotFound:
            logger.error('Error\n\n', exc_info=True)

    def process_download(self):
        """
        Requests metadata file to define if update is needed for tile
        Launches multithread download
        """
        tiles_to_update = self.request_google_cloud_storage_for_latest_acquisition(self.tiles_and_uris_dict)
        self.launch_download_pool(tiles_to_update)

    @staticmethod
    def get_tile_uri(tile_name):
        """
        Reads config and extract name of the tiles which are needed for application
        Converts Tile Name to part of the URI in format [UTM_ZONE]/[LATITUDE_BAND]/[GRID_SQUARE]
        Creates URI for full tile path
        :tile_name: str
        :return: str: tile_uri
        """
        try:
            match = re.search(r'\b\d+', tile_name)
            utm_zone = int(match[0]) if match else None
            if not utm_zone or 1 > utm_zone or utm_zone > 60:
                raise TillNameError(tile_name)
            string = tile_name.replace(match[0], '')

            match = re.search(r'\b[C-Z]', string)
            latitude_band = match[0] if match else None
            if not latitude_band:
                raise TillNameError(tile_name)
            string = string.replace(match[0], '', 1)

            match = re.search(r'\b[A-Z][A-Z]\b', string)
            grid_square = match[0] if match else None
            if not grid_square:
                logger.info(string)
                raise TillNameError(tile_name)
            return f'{utm_zone}/{latitude_band}/{grid_square}'
        except TillNameError:
            logger.error('Error\n\n', exc_info=True)

    def launch_download_pool(self, tiles_to_update):
        """
        For each tile starts own thread processing
        :param tiles_to_update:
        :return:
        """
        with ThreadPoolExecutor(max_workers=settings.MAX_WORKERS) as executor:
            future_list = []
            for tile_name, tile_path in tiles_to_update.items():
                future = executor.submit(self.download_images_from_tiles, tile_name, tile_path)
                future_list.append(future)

        for future in as_completed(future_list):
            if not future.result():
                exit(1)
            else:
                logger.info(f'images for {future.result()[0]} were downloaded')

    def download_images_from_tiles(self, tile_name, tile_path):
        """
        Iterates over folders to fetch tile images folder
        Downloads band specified in config
        :param tile_name:
        :param tile_path:
        :return:
        """
        tile_prefix = f'{tile_path}/IMG_DATA/R20m/'
        blobs = self.storage_bucket.list_blobs(prefix=tile_prefix)

        for blob in blobs:
            band, download_needed = self.file_need_to_be_downloaded(blob.name)
            if download_needed:
                filename = settings.DOWNLOADED_IMAGES_DIR / f'{tile_name}_{band}.jp2'
                self.download_file_from_storage(blob, filename)

        tile_prefix = f'{tile_path}/IMG_DATA/R10m/'
        blobs = self.storage_bucket.list_blobs(prefix=tile_prefix)

        for blob in blobs:
            band, download_needed = self.file_need_to_be_downloaded(blob.name)
            if download_needed:
                filename = settings.DOWNLOADED_IMAGES_DIR / f'{tile_name}_{band}.jp2'
                self.download_file_from_storage(blob, filename)


        tile_prefix = f'{tile_path}/QI_DATA/'
        blobs = self.storage_bucket.list_blobs(prefix=tile_prefix)

        endswith = 'MSK_CLDPRB_20m.jp2'
        for blob in blobs:
            if blob.name.endswith(endswith):
                filename = settings.DOWNLOADED_IMAGES_DIR / f"{tile_name}_{blob.name.split('/')[-1]}"
                self.download_file_from_storage(blob, filename)

        return tile_name, tile_path

    def file_need_to_be_downloaded(self, name):
        """
        Checks if blob is eligible for download through formats specified in config
        :param name:
        :return:
        """
        for band in self.bands_to_download:
            if name.endswith(f'_{band}_10m.jp2') or name.endswith(f'_{band}_20m.jp2'):
                return band, True
        return None, False

    def request_google_cloud_storage_for_latest_acquisition(self, tiles_path):
        """
        Iterates over tile sets and picks latest tile dataset.
        Defines if tile is needed to be updated.
        :param tiles_path:
        :return:
        """
        tiles_to_be_downloaded = {}
        metadata_file = 'MTD_TL.xml'

        for tile_name, tile_path in tiles_path.items():
            logger.info(f'TILE NAME: {tile_name}')
            delimiter = '/'
            tile_uri = f'L2/tiles/{tile_path}/'

            blobs = self.storage_client.list_blobs(self.storage_bucket, prefix=tile_uri, delimiter=delimiter)
            blobs._next_page()

            prefixes = list(blobs.prefixes)
            if not prefixes:
                raise NotFound(f'no such tile_uri: {tile_uri}')
            prefixes.sort(key=self.get_folders_date, reverse=True)
            granule_id_list = [prefix for prefix in prefixes if date_current_test in prefix or date_previous_test in prefix]

            granule_num = 0
            for granule_id in granule_id_list:
                if granule_num < self.sequential_dates_count:
                    prefix = f'{granule_id}GRANULE/'
                    blobs = self.storage_client.list_blobs(self.storage_bucket, prefix=prefix, delimiter=delimiter)
                    blobs._next_page()

                    nested_granule_id_list = list(blobs.prefixes)
                    if not nested_granule_id_list:
                        raise NotFound(f'no such prefix: {prefix}')

                    nested_granule_id = nested_granule_id_list[0]
                    updated_tile_uri = nested_granule_id[:-1] if nested_granule_id.endswith('/') else nested_granule_id

                    filename = settings.DOWNLOADED_IMAGES_DIR / f'{tile_name}_{metadata_file}'
                    print(granule_num, filename)
                    blob = self.storage_bucket.get_blob(f'{updated_tile_uri}/{metadata_file}')
                    if not blob:
                        raise NotFound(f'not found {updated_tile_uri}/{metadata_file}')

                    update_needed = self.define_if_tile_update_needed(blob, f'{tile_name}_{granule_num}', filename)
                    if update_needed:
                        logger.info(f'Tile {tile_name}_{granule_num} will be downloaded from {updated_tile_uri}')
                        tiles_to_be_downloaded[f'{tile_name}_{granule_num}'] = f'{updated_tile_uri}'
                        os.remove(filename)
                        granule_num += 1
                else:
                    break
        print(tiles_to_be_downloaded)
        return tiles_to_be_downloaded

    def define_if_tile_update_needed(self, blob, tile_name, filename) -> bool:
        """
        Checks hash of the metadata file for latest image and information from DB
        Downloads metadata file if hash is not equal
        Checks cloud coverage value from metadata file if it lower then one from settings - allows to download images
        :param blob:
        :param tile_name:
        :param filename:
        :return:
        """
        filename = str(filename)
        self.download_file_from_storage(blob, filename)
        nodata_pixel_value = self.define_xml_node_value(filename, 'NODATA_PIXEL_PERCENTAGE')
        # print('====== NO DATA PIXEL VALUE ======')
        # print(nodata_pixel_value)
        if nodata_pixel_value >= settings.MAXIMUM_EMPTY_PIXEL_PERCENTAGE:
            return False
        cloud_coverage_value = self.define_xml_node_value(filename, 'CLOUDY_PIXEL_PERCENTAGE')
        # print('====== CLOUD COVERAGE VALUE ======')
        # print(cloud_coverage_value)
        if cloud_coverage_value <= settings.MAXIMUM_CLOUD_PERCENTAGE_ALLOWED:
            return True
        else:
            return False

    def get_folders_date(self, path):
        match = re.search(r'_\d{8}T\d{6}', path)
        if match:
            return match[0]
        else:
            raise ValueError(f'No date in {self.bucket_name} for {path}')

    @staticmethod
    def download_file_from_storage(blob, filename):
        """
        Downloads blob to local storage
        :param blob:
        :param filename:
        :return:
        """
        with open(filename, 'wb') as new_file:
            blob.download_to_file(new_file)

    @staticmethod
    def define_xml_node_value(xml_file_name, node):
        """
        Parsing XML file for passed node name
        :param xml_file_name:
        :param node:
        :return:
        """
        xml_dom = minidom.parse(xml_file_name)
        try:
            xml_node = xml_dom.getElementsByTagName(node)
            xml_node_value = xml_node[0].firstChild.data
            return float(xml_node_value)
        except FileNotFoundError(f'No such file: {xml_file_name}'):
            logger.error('Error\n\n', exc_info=True)
        except ParseError(f'no such node ({node}) in the {xml_file_name}'):
            logger.error('Error\n\n', exc_info=True)
            return None


sentinel_downloader = SentinelDownload()
sentinel_downloader.process_download()
