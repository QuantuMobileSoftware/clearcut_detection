import os

from configparser import ConfigParser
from datetime import datetime

def path_exists_or_create(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

config_test = ConfigParser(allow_no_value=True)
config_test.read('test_config.ini')

area_tile_set_test = set(config_test.get('config', 'AREA_TILE_SET').split())
bands_to_download = config_test.get('config', 'BANDS_TO_DOWNLOAD').split()
date_current_test = config_test.get('config', 'DATE_CURRENT')
date_previous_test = config_test.get('config', 'DATE_PREVIOUS')

TEST_PATH = 'data/test_data'
TEST_POLYGONS = f'{TEST_PATH}/36UYA_test_data.geojson'
TEST_TILES = f'{TEST_PATH}/tiles_to_download.txt'
DATE_CURRENT = datetime.strptime(date_current_test, '%Y%m%d')
DATE_PREVIOUS = datetime.strptime(date_previous_test, '%Y%m%d')

# Target metrics values
GOLD_STANDARD_F1SCORES = f'{TEST_PATH}/gold_standard.txt'
GOLD_DICE = 0.2811
GOLD_IOU = 0.2545
IOU_THRESHOLD = 0.3
SUCCESS_THRESHOLD = 0.05
