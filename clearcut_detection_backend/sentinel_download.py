import os
import csv
import datetime
import subprocess

from django.conf import settings
from concurrent.futures import ThreadPoolExecutor
from configparser import ConfigParser
from google.cloud import storage
from xml.dom import minidom

from clearcuts.models import TileInformation
from utils import path_exists_or_create, Bands

DOWNLOADED_IMAGES_DIR = path_exists_or_create('data/source_images/')


class SentinelDownload:

    def __init__(self):
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = './key.json'
        self.tile_dates_count = settings.MAXIMUM_DATES_REVIEWED_FOR_TILE
        self.sequential_dates_count = settings.MAXIMUM_DATES_STORE_FOR_TILE
        self.storage_client = storage.Client()
        self.storage_bucket = self.storage_client.get_bucket('gcp-public-data-sentinel-2')
        self.config = ConfigParser(allow_no_value=True)
        self.config.read('gcp_config.ini')
        self.area_tile_set = self.config.get('config', 'AREA_TILE_SET').split()
        self.bands_to_download = self.config.get('config', 'BANDS_TO_DOWNLOAD').split()
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
        :param tile_info:
        :return:
        """
        tile_prefix = f'{tile_path}/IMG_DATA/R20m/'
        blobs = self.storage_bucket.list_blobs(prefix=tile_prefix)

        for blob in blobs:
            band, download_needed = self.file_need_to_be_downloaded(blob.name)
            if download_needed:
                filename = os.path.join(DOWNLOADED_IMAGES_DIR, f'{tile_name}_{band}.jp2')
                self.download_file_from_storage(blob, filename)
                tile_info = TileInformation.objects.get(tile_name=tile_name)
                if band == Bands.B11.value:
                    tile_info.source_b11_location = filename
                elif band == Bands.B12.value:
                    tile_info.source_b12_location = filename
                elif band == Bands.B8A.value:
                    tile_info.source_b8a_location = filename
                else:
                    continue
                tile_info.save()
        
        tile_prefix = f'{tile_path}/IMG_DATA/R10m/'
        blobs = self.storage_bucket.list_blobs(prefix=tile_prefix)

        for blob in blobs:
            band, download_needed = self.file_need_to_be_downloaded(blob.name)
            if download_needed:
                filename = os.path.join(DOWNLOADED_IMAGES_DIR, f'{tile_name}_{band}.jp2')
                self.download_file_from_storage(blob, filename)
                tile_info = TileInformation.objects.get(tile_name=tile_name)
                if band == Bands.B04.value:
                    tile_info.source_b04_location = filename
                elif band == Bands.B08.value:
                    tile_info.source_b08_location = filename
                elif band == Bands.TCI.value:
                    tile_info.source_tci_location = filename
                else:
                    continue
                tile_info.save()

        tile_prefix = f'{tile_path}/QI_DATA/' #MSK_CLDPRB_20m.jp2
        blobs = self.storage_bucket.list_blobs(prefix=tile_prefix)
        for blob in blobs:
            if blob.name.endswith('MSK_CLDPRB_20m.jp2'):
                tile_info = TileInformation.objects.get(tile_name=tile_name)
                filename = os.path.join(DOWNLOADED_IMAGES_DIR, f"{tile_name}_{blob.name.split('/')[-1]}")
                self.download_file_from_storage(blob, filename)
                tile_info.source_clouds_location = filename
                tile_info.save()
        #print(TileInformation.objects.values())
        #print(list(TileInformation.objects.values('tile_index').distinct()))

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
        for tile_name, tile_path in tiles_path.items():
            print('====== TILE NAME ======')
            print(tile_name)
            base_uri = 'gs://gcp-public-data-sentinel-2'
            tile_uri = f'L2/tiles/{tile_path}'
            metadata_file = 'MTD_TL.xml'
            command = f'gsutil ls -l {base_uri}/{tile_uri}/ | sort -k2n | tail -n{self.tile_dates_count}'
            granule_id_list = self.find_granule_id(command)
            granule_id_list.reverse()
            granule_num = 0
            for granule_id in granule_id_list:
                if granule_num < self.sequential_dates_count:
                    print('====== GRANULE ID ======')
                    print(granule_id)
                    nested_command = f'gsutil ls -l {base_uri}/{tile_uri}/{granule_id}/GRANULE/ | sort -k2n | tail -n1'
                    nested_granule_id_list = self.find_granule_id(nested_command)
                    nested_granule_id = nested_granule_id_list[0]
                    updated_tile_uri = f'{tile_uri}/{granule_id}/GRANULE/{nested_granule_id}'
                    filename = os.path.join(DOWNLOADED_IMAGES_DIR, f'{tile_name}_{metadata_file}')
                    try:
                        blob = self.storage_bucket.get_blob(f'{updated_tile_uri}/{metadata_file}')
                        update_needed = self.define_if_tile_update_needed(blob, f'{tile_name}_{granule_num}', filename)
                        print('====== IS UPDATE NEEDED ======')
                        print(update_needed)
                        if update_needed:
                            tiles_to_be_downloaded[f'{tile_name}_{granule_num}'] = f'{updated_tile_uri}'
                            os.remove(filename)
                            granule_num+=1
                    except Exception as e:
                        print(e)
                        print(dir(e))
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
        
        tile_info, created = TileInformation.objects.get_or_create(tile_name=tile_name)

        if not created:
            update_needed = blob.md5_hash != tile_info.tile_metadata_hash
            if not update_needed:
                return False

        self.download_file_from_storage(blob, filename)
        nodata_pixel_value = self.define_xml_node_value(filename, 'NODATA_PIXEL_PERCENTAGE')
        print('====== NO DATA PIXEL VALUE ======')
        print(nodata_pixel_value)
        if nodata_pixel_value >= settings.MAXIMUM_EMPTY_PIXEL_PERCENTAGE:
            return False
        cloud_coverage_value = self.define_xml_node_value(filename, 'CLOUDY_PIXEL_PERCENTAGE')
        print('====== CLOUD COVERAGE VALUE ======')
        print(cloud_coverage_value)
        if cloud_coverage_value <= settings.MAXIMUM_CLOUD_PERCENTAGE_ALLOWED:
            tile_info.cloud_coverage = cloud_coverage_value
            tile_info.tile_metadata_hash = blob.md5_hash
            tile_info.tile_index = tile_name.split('_')[0]
            tile_info.tile_date = datetime.datetime.strptime(blob.name.split('_')[-2][:8], '%Y%m%d')
            tile_info.save()
            return True
        else:
            return False

    def find_granule_id(self, command):
        """
        Requests next GCS folder node.
        [:-9] is used to cut out _$folder$ from naming
        :param command:
        :return:
        """
        # print(command)
        granule_id_list = []
        process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
        process_stdout = process.communicate()[0].decode('utf8')
        for tile_output in process_stdout.strip().split('\n'):
            tile_output_path = tile_output.strip().split()[-1]
            if tile_output_path.endswith('_$folder$'):
                tile_output = tile_output_path[:-9].split('/')[-1]
                granule_id_list.append(tile_output)
            else:
                granule_id_list = tile_output_path.split('/')[-1]

        return granule_id_list

    def download_file_from_storage(self, blob, filename):
        """
        Downloads blob to local storage
        :param blob:
        :param filename:
        :return:
        """
        with open(filename, 'wb') as new_file:
            blob.download_to_file(new_file)

    def define_xml_node_value(self, xml_file, node):
        """
        Parsing XML file for passed node name
        :param xml_file:
        :param node:
        :return:
        """
        xml_dom = minidom.parse(xml_file)
        try:
            xml_node = xml_dom.getElementsByTagName(node)
            xml_node_value = xml_node[0].firstChild.data
            return float(xml_node_value)
        except Exception as e:
            print(e)
            return None
