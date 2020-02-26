import logging
import os
import subprocess

from django.conf import settings
from concurrent.futures import ThreadPoolExecutor
from configparser import ConfigParser
from google.cloud import storage
from xml.dom import minidom

from clearcuts.models import TileInformation

logger = logging.getLogger(__name__)
DATA_DIR = 'data'


class SentinelDownload:

    def __init__(self):
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = './key.json'
        self.storage_client = storage.Client()
        self.storage_bucket = self.storage_client.get_bucket('gcp-public-data-sentinel-2')
        self.config = ConfigParser(allow_no_value=True)
        self.config.read('gcp_config.ini')
        self.area_tile_set = self.config.get('config', 'KHARKIV_OBLAST_TILE_SET').split()
        self.bands_to_download = self.config.get('config', 'BANDS_TO_DOWNLOAD').split()
        self.resolution = self.config.get('config', 'RESOLUTION')
        self.executor = ThreadPoolExecutor(max_workers=10)

    def process_download(self):
        """
        Creates URI parts out of tile names
        Requests metadata file to define if update is needed for tile
        Launches multithread download
        """
        tiles_and_uris_dict = self.tile_uri_composer()
        tiles_to_update = self.request_google_cloud_storage_for_latest_acquisition(tiles_and_uris_dict)
        self.launch_download_pool(tiles_to_update)

    def tile_uri_composer(self):
        """
        Reads config and extract name of the tiles which are needed for application
        Converts Tile Name to part of the URI in format [UTM_ZONE]/[LATITUDE_BAND]/[GRID_SQUARE]
        Creates URI for full tile path
        :return:
        """
        tile_location_uri_part_list = \
            {tile_name: f'{tile_name[:2]}/{tile_name[2:3]}/{tile_name[3:]}' for tile_name in self.area_tile_set}

        return tile_location_uri_part_list

    def launch_download_pool(self, tiles_to_update):
        """
        For each tile starts own thread processing
        :param tiles_to_update:
        :return:
        """
        for tile_name, tile_path in tiles_to_update.items():
            threads = self.executor.submit(self.download_images_from_tiles, tile_name, tile_path)

    def download_images_from_tiles(self, tile_name, tile_path):
        """
        Iterates over folders to fetch tile images folder
        Downloads band specified in config
        :param tile_name:
        :param tile_path:
        :return:
        """
        command = f'gsutil ls -l {tile_path}/GRANULE/ | sort -k2n | tail -n1'
        granule_id = self.find_granule_id(command)
        decomposed_tile_uri = tile_path.split('/')
        tile_prefix_start = '/'.join(decomposed_tile_uri[3:])
        tile_prefix = f'{tile_prefix_start}/GRANULE/{granule_id}/IMG_DATA/R{self.resolution}/'
        blobs = self.storage_bucket.list_blobs(prefix=tile_prefix)
        # check_blob = lambda value: any([value.endswith(f'_{band}_{self.resolution}.jp2')
        #                                 for band in self.bands_to_download])
        for blob in blobs:
            band, download_needed = self.file_need_to_be_downloaded(blob.name)
            if download_needed:
                filename = os.path.join(DATA_DIR, f'{tile_name}_{band}.jp2')
                self.download_file_from_storage(blob, filename)
                tile_info = TileInformation.objects.get(tile_name=tile_name)
                tile_info.tile_location = filename
                tile_info.save()

    def file_need_to_be_downloaded(self, name):
        """
        Checks if blob is eligible for download through formats specified in config
        :param name:
        :return:
        """
        for band in self.bands_to_download:
            if name.endswith(f'_{band}_{self.resolution}.jp2'):
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
        for tile_name, tile_path in tiles_path.items():
            base_uri = 'gs://gcp-public-data-sentinel-2'
            tile_uri = f'L2/tiles/{tile_path}'
            metadata_file = 'MTD_MSIL2A.xml'
            command = f'gsutil ls -l {base_uri}/{tile_uri}/ | sort -k2n | tail -n1'
            granule_id = self.find_granule_id(command)

            blob = self.storage_bucket.get_blob(f'{tile_uri}/{granule_id}/{metadata_file}')
            update_needed = self.define_if_tile_update_needed(blob, tile_name, metadata_file)
            if update_needed:
                tiles_to_be_downloaded[f'{tile_name}'] = f'{base_uri}/{tile_uri}/{granule_id}'

        return tiles_to_be_downloaded

    def define_if_tile_update_needed(self, blob, tile_name, metadata_file) -> bool:
        """
        Checks hash of the metadata file for latest image and information from DB
        Downloads metadata file if hash is not equal
        Checks cloud coverage value from metadata file if it lower then one from settings - allows to download images
        :param blob:
        :param tile_name:
        :param metadata_file:
        :return:
        """
        update_status = False
        tile_info, created = TileInformation.objects.get_or_create(tile_name=tile_name)
        if not created:
            update_needed = blob.md5_hash != tile_info.tile_metadata_hash
            if update_needed:
                filename = os.path.join(DATA_DIR, '{tile_name}_{metadata_file}')
                self.download_file_from_storage(blob, filename)
                cloud_coverage_value = self.define_cloud_coverage_value(filename)
                if float(cloud_coverage_value) <= settings.MAXIMUM_CLOUD_PERCENTAGE_ALLOWED:
                    update_status = True
                    tile_info.cloud_coverage = cloud_coverage_value
                    tile_info.tile_metadata_hash = blob.md5_hash
                    tile_info.save()
        else:
            filename = os.path.join(DATA_DIR, f'{tile_name}_{metadata_file}')
            self.download_file_from_storage(blob, filename)
            cloud_coverage_value = self.define_cloud_coverage_value(filename)
            update_status = True
            tile_info.cloud_coverage = cloud_coverage_value
            tile_info.tile_metadata_hash = blob.md5_hash
            tile_info.save()
            os.remove(filename)

        return update_status

    def find_granule_id(self, command):
        """
        Requests next GCS folder node.
        [:-9] is used to cut out _$folder$ from naming
        :param command:
        :return:
        """
        process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
        process_stdout = process.communicate()[0].strip()
        stdout_string = process_stdout.decode('utf8').split()[-1]
        stdout_string_path = stdout_string.split('/')[-1]
        if stdout_string_path.endswith('_$folder$'):
            granule_id = stdout_string_path[:-9]
        else:
            granule_id = stdout_string_path

        return granule_id

    def download_file_from_storage(self, blob, filename):
        """
        Downloads blob to local storage
        :param blob:
        :param filename:
        :return:
        """
        with open(filename, 'wb') as new_file:
            blob.download_to_file(new_file)

    def define_cloud_coverage_value(self, xml_file):
        """
        Parsing XML file for HIGH_PROBA_CLOUDS_PERCENTAGE value
        :param xml_file:
        :return:
        """
        xml_dom = minidom.parse(xml_file)
        try:
            high_proba_cloud_node = xml_dom.getElementsByTagName('HIGH_PROBA_CLOUDS_PERCENTAGE')
            cloud_percentage_value = high_proba_cloud_node[0].firstChild.data
            return cloud_percentage_value
        except Exception as e:
            print(e)
            return None
