import json
import os
import logging
import requests
import yaml
from concurrent.futures import ThreadPoolExecutor

from test.settings import MODEL_TIFFS_DIR, MAX_WORKERS, DATA_DIR
from test.utils import area_tile_set_test, path_exists_or_create
from test.utils import download_file_from_google_drive
from test.utils import gdrive_ids

model_call_config = 'model_call_config.yml'
logger = logging.getLogger('model_call')

def file_download():
    tile = area_tile_set_test.pop()

    test_tile_path = path_exists_or_create(f'{MODEL_TIFFS_DIR}/{tile}')
    test_tile_path = {}

    test_tile_path['current'] = path_exists_or_create(f'{MODEL_TIFFS_DIR}/{tile}/{tile}_0/') + f'{tile}_0.tif'
    test_tile_path['previous'] = path_exists_or_create(f'{MODEL_TIFFS_DIR}/{tile}/{tile}_1/') + f'{tile}_1.tif'

    test_tile_path['cloud_current'] = path_exists_or_create(f'{MODEL_TIFFS_DIR}/{tile}/{tile}_0/') + 'clouds.tiff'
    test_tile_path['cloud_previous'] = path_exists_or_create(f'{MODEL_TIFFS_DIR}/{tile}/{tile}_1/') + 'clouds.tiff'

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for order in test_tile_path.keys():
            if not os.path.exists(test_tile_path[order]):
                executor.submit(download_file_from_google_drive, gdrive_ids[order], test_tile_path[order])
    
    print(test_tile_path)
    return test_tile_path
    

def model_predict():
    test_tile_path = file_download()
    tif_path = "/".join(test_tile_path['current'].split('/')[:4])
    logger.info(f'raster_prediction {tif_path}')
    print(f'raster_prediction {tif_path}')
    results = raster_prediction(tif_path)
    logger.info(f'results:\n{results}')
    print(f'results:\n{results}')
    results_path = os.path.join(DATA_DIR, results[0].get('polygons'))
    return results, test_tile_path

# TODO: add docstring
def raster_prediction(tif_path):
    with open(model_call_config, 'r') as config:
        cfg = yaml.load(config, Loader=yaml.SafeLoader)
    model_api_cfg = cfg["model-api"]
    api_endpoint = "http://{host}:{port}/{endpoint}".format(
        host=model_api_cfg["host"],
        port=model_api_cfg["port"],
        endpoint=model_api_cfg["endpoint"]
    )
    data = {"image_path": tif_path}
    logger.info(f'sending request to model API for\n{tif_path}')
    try:
        response = requests.post(url=api_endpoint, json=data)
        result = response.text
        datastore = json.loads(result)
        return datastore
    except (ValueError, Exception):
        logger.error('Error\n\n', exc_info=True)
