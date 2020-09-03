import os
import re
import datetime
import logging
from enum import Enum
from distutils.util import strtobool
from django.conf import settings
from django.core.exceptions import ObjectDoesNotExist
from concurrent.futures import ThreadPoolExecutor, as_completed
from google.cloud import storage
from google.cloud.exceptions import NotFound
from xml.dom import minidom
from xml.etree.ElementTree import ParseError
from clearcuts.models import Tile
# from services.configuration import bands_to_download
from downloader.models import SourceJp2Images as Sjp

force_download_img = strtobool(os.environ.get('FORCE_DOWNLOAD_IMG', 'false'))

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

    def __init__(self, tile_index):
        settings.DOWNLOADED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = './key.json'
        self.tile_index = tile_index
        # self.tile_dates_count = settings.MAXIMUM_DATES_REVIEWED_FOR_TILE
        # self.sequential_dates_count = settings.MAXIMUM_DATES_STORE_FOR_TILE
        # self.bands_to_download = bands_to_download  # TODO to db
        self.bucket_name = 'gcp-public-data-sentinel-2'  # TODO to settings
        self.prefix = 'L2/tiles'
        self.metadata_file = 'MTD_TL.xml'
        # self.tiles_and_uris_dict = {tile_name: self.get_tile_uri(tile_name) for tile_name in self.area_tile_set}
        self.storage_client = storage.Client()
        self.storage_bucket = self.get_storage_bucket()

        logger.info(f'area tile set for download: {self.tile_index}')
        # logger.info(f'bands to download:{self.bands_to_download}')

    def get_storage_bucket(self):
        try:
            return self.storage_client.get_bucket(self.bucket_name)
        except NotFound:
            logger.error(f'Error, cant find bucket {self.bucket_name} \n\n', exc_info=True)

    def process_download(self):
        """
        Requests metadata file to define if update is needed for tile
        Launches multi thread download
        """
        self.request_google_cloud_storage_for_historical_data()
        self.launch_download_pool()

    @staticmethod
    def get_tile_uri(tile_name):
        """
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

    def launch_download_pool(self):
        """
        For each tile_date starts own thread processing
        :return:
        """
        with ThreadPoolExecutor(max_workers=settings.MAX_WORKERS*4) as executor:
            for source_jp2_images in Sjp.objects.select_related('tile').filter(
                    tile__tile_index=self.tile_index,
                    is_new=1
            ).order_by('image_date'):
                executor.submit(self.download_images_from_tiles, source_jp2_images)

    def r_10m_download(self, source_jp2_images):
        tile_uri = source_jp2_images.tile_uri
        image_date = source_jp2_images.image_date
        tile_name = source_jp2_images.tile.tile_index
        file_path = settings.DOWNLOADED_IMAGES_DIR / tile_name / str(image_date)
        tile_prefix = f'{tile_uri}/IMG_DATA/R10m/'
        blobs = self.storage_bucket.list_blobs(prefix=tile_prefix)

        is_blob = False
        for blob in blobs:
            is_blob = True
            logger.info(blob.name)

            filename = None

            img_name = blob.name.split('/')[-1]
            logger.info(img_name)

            if img_name.endswith('_B04_10m.jp2'):
                filename = file_path / 'B04_10m.jp2'
                source_jp2_images.source_b04_location = filename
                source_jp2_images.save()
            if img_name.endswith('_B08_10m.jp2'):
                filename = file_path / 'B08_10m.jp2'
                source_jp2_images.source_b08_location = filename
                source_jp2_images.save()
            if img_name.endswith('_TCI_10m.jp2'):
                filename = file_path / 'TCI_10m.jp2'
                source_jp2_images.source_tci_location = filename
                source_jp2_images.save()

            logger.info(f'filename: {filename}')
            if filename and not filename.is_file() or filename and force_download_img:
                try:
                    self.download_file_from_storage(blob, filename)
                    source_jp2_images.is_downloaded += 1
                except (ValueError, Exception):
                    source_jp2_images.is_downloaded = -1

                    logger.error(
                        f'{__name__}.{self.download_images_from_tiles.__qualname__} Error in uri: {blob.name}')

                source_jp2_images.save()

            elif filename and filename.is_file():
                source_jp2_images.is_downloaded += 1
                source_jp2_images.save()

        if not is_blob:
            source_jp2_images.is_downloaded = -1
            source_jp2_images.save()
            logger.error(f'{__name__}.{self.download_images_from_tiles.__qualname__} Error in uri: {tile_prefix}')
            return

    def r_20m_download(self, source_jp2_images):

        tile_uri = source_jp2_images.tile_uri
        image_date = source_jp2_images.image_date
        tile_name = source_jp2_images.tile.tile_index
        file_path = settings.DOWNLOADED_IMAGES_DIR / tile_name / str(image_date)
        tile_prefix = f'{tile_uri}/IMG_DATA/R20m/'
        blobs = self.storage_bucket.list_blobs(prefix=tile_prefix)

        is_blob = False
        for blob in blobs:
            is_blob = True
            logger.info(blob.name)
            filename = None

            img_name = blob.name.split('/')[-1]
            logger.info(img_name)
            if img_name.endswith('_B11_20m.jp2'):
                filename = file_path / 'B11_20m.jp2'
                source_jp2_images.source_b11_location = filename
                source_jp2_images.save()
            if img_name.endswith('_B12_20m.jp2'):
                filename = file_path / 'B12_20m.jp2'
                source_jp2_images.source_b12_location = filename
                source_jp2_images.save()
            if img_name.endswith('_B8A_20m.jp2'):
                filename = file_path / 'B8A_20m.jp2'
                source_jp2_images.source_b8a_location = filename
                source_jp2_images.save()

            if filename and not filename.is_file() or filename and force_download_img:
                try:
                    self.download_file_from_storage(blob, filename)
                    source_jp2_images.is_downloaded += 1
                except (ValueError, Exception):
                    source_jp2_images.is_downloaded = -1

                    logger.error(
                        f'{__name__}.{self.download_images_from_tiles.__qualname__} Error in uri: {blob.name}')

                source_jp2_images.save()

            elif filename and filename.is_file():
                source_jp2_images.is_downloaded += 1
                source_jp2_images.save()

        if not is_blob:
            source_jp2_images.is_downloaded = -1
            source_jp2_images.save()
            logger.error(f'{__name__}.{self.download_images_from_tiles.__qualname__} Error in uri: {tile_prefix}')
            return

    def qi_data_download(self, source_jp2_images):
        tile_uri = source_jp2_images.tile_uri
        image_date = source_jp2_images.image_date
        tile_name = source_jp2_images.tile.tile_index
        file_path = settings.DOWNLOADED_IMAGES_DIR / tile_name / str(image_date)
        tile_prefix = f'{tile_uri}/QI_DATA/'
        blobs = self.storage_bucket.list_blobs(prefix=tile_prefix)

        is_blob = False
        for blob in blobs:
            is_blob = True
            logger.info(blob.name)
            filename = None

            img_name = blob.name.split('/')[-1]
            logger.info(img_name)
            if img_name == 'MSK_CLDPRB_20m.jp2':
                filename = file_path / 'MSK_CLDPRB_20m.jp2'
                source_jp2_images.source_clouds_location = filename
                source_jp2_images.save()

            if filename and not filename.is_file() or filename and force_download_img:
                try:
                    self.download_file_from_storage(blob, filename)
                    source_jp2_images.is_downloaded += 1
                except (ValueError, Exception):
                    source_jp2_images.is_downloaded = -1

                    logger.error(
                        f'{__name__}.{self.download_images_from_tiles.__qualname__} Error in uri: {blob.name}')

                source_jp2_images.save()

            elif filename and filename.is_file():
                source_jp2_images.is_downloaded += 1
                source_jp2_images.save()

        if not is_blob:
            source_jp2_images.is_downloaded = -1
            source_jp2_images.save()
            logger.error(f'{__name__}.{self.download_images_from_tiles.__qualname__} Error in uri: {tile_prefix}')
            return

    def download_images_from_tiles(self, source_jp2_images):
        """
        Downloads band specified in config
        :param source_jp2_images:
        :return:
        """
        source_jp2_images.is_downloaded = 0
        source_jp2_images.save()
        self.r_20m_download(source_jp2_images)
        self.r_10m_download(source_jp2_images)
        self.qi_data_download(source_jp2_images)

        logger.info(f'finish for {source_jp2_images.image_date}')

    def request_google_cloud_storage_for_new_data(self):
        """
        Iterates over tile blobs .
        Select tile uri by:
         date - it must be greater than tile.last_date stored in db,
         clouds coverage and nodata_pixel and store it to db.
        :return:
        """
        prefixes = self.get_prefixes_list_by_tile_name(self.tile_index)

        tile = Tile.objects.get(tile_index=self.tile_index)
        if tile.last_date is not None:
            granule_id_list = []
            prefixes.sort(key=self.get_granule_date, reverse=True)
            for prefix in prefixes:
                if self.get_granule_date(prefix) > tile.last_date or self.get_granule_date(prefix) == tile.last_date:
                    granule_id_list.append(prefix)

            self.granule_id_list_to_db(self.tile_index, granule_id_list)

    def request_google_cloud_storage_for_historical_data(self):
        """
        Iterates over tile blobs.
        Select tile uri by clouds coverage and nodata_pixel and store it to db.
        :return:
        """
        prefixes = self.get_prefixes_list_by_tile_name(self.tile_index)

        prefixes.sort(key=self.get_granule_date)
        granule_id_list = []
        for prefix in prefixes:
            if str(self.get_granule_date(prefix)) > settings.START_DATE_FOR_SCAN:
                granule_id_list.append(prefix)

        self.granule_id_list_to_db(self.tile_index, granule_id_list)

    def get_prefixes_list_by_tile_name(self, tile_name):
        """
        get prefix list for tile_name from bucket
        :param tile_name: str
        :return: list
        """
        tile_path = self.get_tile_uri(tile_name)

        tile_uri = f'L2/tiles/{tile_path}/'

        blobs = self.storage_client.list_blobs(self.storage_bucket, prefix=tile_uri, delimiter='/')
        blobs._next_page()

        prefixes = list(blobs.prefixes)
        if not prefixes:
            raise NotFound(f'no such tile_uri: {tile_uri}')
        return prefixes

    def granule_id_list_to_db(self, tile_name, granule_id_list):
        """
        Iterates over granule_id_list, fetch meta information and store it to db
        :param tile_name: str
        :param granule_id_list: list
        :return:
        """
        for granule_id in granule_id_list:
            granule_date = self.get_granule_date(granule_id)
            logger.info(f'granule_date: {granule_date}')
            prefix = f'{granule_id}GRANULE/'
            blobs = self.storage_client.list_blobs(self.storage_bucket, prefix=prefix, delimiter='/')
            blobs._next_page()

            nested_granule_id_list = list(blobs.prefixes)
            if not nested_granule_id_list:
                raise NotFound(f'no such prefix: {prefix}')

            nested_granule_id = nested_granule_id_list[0]
            updated_tile_uri = nested_granule_id[:-1] if nested_granule_id.endswith('/') else nested_granule_id

            blob = self.storage_bucket.get_blob(f'{updated_tile_uri}/{self.metadata_file}')
            if not blob:
                raise NotFound(f'not found {updated_tile_uri}/{self.metadata_file}')

            self.add_source_to_db(blob, tile_name, self.metadata_file, updated_tile_uri)

    @staticmethod
    def get_date_from_uri(tile_uri):
        """
        get date from uri string
        :param tile_uri: str
        :return: date
        """
        date_time_str = tile_uri.split('_')[-1]
        logger.info(f'date_time_str: {date_time_str}')
        date_time_obj = datetime.datetime.strptime(date_time_str, '%Y%m%dT%H%M%S')
        return date_time_obj.date()

    def add_source_to_db(self, blob, tile_name, metadata_file, tile_uri):
        """
        Checks NODATA_PIXEL_PERCENTAGE and CLOUDY_PIXEL_PERCENTAGE value from metadata file.
        If it lower then one from settings - store information about files to db and return true
        :param blob:
        :param tile_name: str
        :param metadata_file: Path or str
        :param tile_uri: str
        :return: bool
        """
        tile_date = self.get_date_from_uri(tile_uri)
        logger.info(f'uri_date: {tile_date}')
        download_img_path = settings.DOWNLOADED_IMAGES_DIR / tile_name / str(tile_date)
        download_img_path.mkdir(parents=True, exist_ok=True)
        filename = download_img_path / metadata_file
        filename_str = str(filename)

        if not filename.is_file():
            self.download_file_from_storage(blob, filename_str)

        nodata_pixel_value = float(self.define_xml_node_value(filename_str, 'NODATA_PIXEL_PERCENTAGE'))
        if nodata_pixel_value >= settings.MAXIMUM_EMPTY_PIXEL_PERCENTAGE:
            return
        cloud_coverage_value = float(self.define_xml_node_value(filename_str, 'CLOUDY_PIXEL_PERCENTAGE'))
        if cloud_coverage_value >= settings.MAXIMUM_CLOUD_PERCENTAGE_ALLOWED:
            return

        tile = Tile.objects.get(tile_index=tile_name)
        try:
            source_jp2_images = Sjp.objects.get(tile=tile, image_date=tile_date)
            source_jp2_images.is_new = 1
            source_jp2_images.save()
            logger.info(f'record for {tile_name}-{source_jp2_images.image_date} was already created')
        except ObjectDoesNotExist:
            source_jp2_images = Sjp.objects.create(
                tile=tile,
                image_date=tile_date,
                tile_uri=tile_uri,
                cloud_coverage=cloud_coverage_value,
                nodata_pixel=nodata_pixel_value,
            )
            source_jp2_images.save()
            logger.info(f'record for {tile_name}-{source_jp2_images.image_date} is created')
        return

    def get_granule_date(self, path):
        match = re.search(r'_\d{8}T\d{6}', path)
        if match:
            date_time_str = match[0][1:]
            date_time_obj = datetime.datetime.strptime(date_time_str, '%Y%m%dT%H%M%S')
            granule_date = date_time_obj.date()
            return granule_date
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
        logger.info(f'start download {filename}')
        with open(filename, 'wb') as new_file:
            blob.download_to_file(new_file)
        logger.info(f'download {filename} finished')
        return

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
            return xml_node_value
        except FileNotFoundError(f'No such file: {xml_file_name}'):
            logger.error('Error\n\n', exc_info=True)
        except ParseError(f'no such node ({node}) in the {xml_file_name}'):
            logger.error('Error\n\n', exc_info=True)
            return None
