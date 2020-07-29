import os
import requests
import urllib.request

from configparser import ConfigParser
from datetime import datetime

from test.settings import DATA_DIR, METRIC_CONFIG, TEST_CONFIG

def download_file_from_google_drive(id, destination):
    # https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)


def path_exists_or_create(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def download_dataset(file_url, save_path):
    urllib.request.urlretrieve(file_url, save_path)
    return save_path

path_exists_or_create(DATA_DIR)

config_test = ConfigParser(allow_no_value=True)
config_test.read(TEST_CONFIG)

area_tile_set_test = set(config_test.get('config', 'AREA_TILE_SET').split())
bands_to_download = config_test.get('config', 'BANDS_TO_DOWNLOAD').split()
date_current_test = config_test.get('config', 'DATE_CURRENT')
date_previous_test = config_test.get('config', 'DATE_PREVIOUS')
test_polys_url = config_test.get('config', 'TEST_POLYGONS_URL')
metrics_url = config_test.get('config', 'METRICS_URL')

gdrive_ids = {}
gdrive_ids['current'] = config_test.get('config', 'TEST_TILE_CURRENT_GDRIVE_ID')
gdrive_ids['previous'] = config_test.get('config', 'TEST_TILE_PREVIOUS_GDRIVE_ID')
gdrive_ids['cloud_current'] = config_test.get('config', 'TEST_CLOUDS_CURRENT_GDRIVE_ID')
gdrive_ids['cloud_previous'] = config_test.get('config', 'TEST_CLOUDS_PREVIOUS_GDRIVE_ID')


TEST_POLYGONS = download_dataset(test_polys_url, f'{DATA_DIR}/test_clearcuts.geojson')
DATE_CURRENT = datetime.strptime(date_current_test, '%Y%m%d')
DATE_PREVIOUS = datetime.strptime(date_previous_test, '%Y%m%d')

# Target metrics values
download_dataset(metrics_url, METRIC_CONFIG)

config_metrics = ConfigParser(allow_no_value=True)
config_metrics.read(METRIC_CONFIG)

GOLD_DICE = float(config_metrics.get('metric', 'GOLD_DICE'))
GOLD_IOU = float(config_metrics.get('metric', 'GOLD_IOU'))
IOU_THRESHOLD = float(float(config_metrics.get('metric', 'IOU_THRESHOLD')))
GOLD_F1SCORE = float(config_metrics.get('metric', 'GOLD_F1SCORE'))
SUCCESS_THRESHOLD = float(config_metrics.get('metric', 'SUCCESS_THRESHOLD'))
