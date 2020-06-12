import json
import os
from concurrent.futures.thread import ThreadPoolExecutor
from shutil import rmtree

import requests
import yaml

from clearcuts.geojson_save import save
from clearcuts.models import TileInformation
from prepare_tif import prepare_tiff


class ModelCaller:
    """
    Class for asynchronous calling of model's API
    """
    def __init__(self):
        self.query = TileInformation.objects.filter(tile_name__isnull=False,
                                                    source_b04_location__isnull=False,
                                                    source_b08_location__isnull=False,
                                                    source_tci_location__isnull=False)
        self.data_dir = 'data'
        self.executor = ThreadPoolExecutor(max_workers=10)

    def start(self):
        for tile in self.query:
            self.executor.submit(self.process, tile)

    def process(self, tile):
        """
        Converting jp2file to tiff, then sending its to model and saving results to db
        """
        prepare_tiff(tile)
        results = raster_prediction(tile.model_tiff_location)

        results_path = os.path.join(self.data_dir, results[0].get('polygons'))
        save(results_path)
        # clean up
        rmtree(os.path.dirname(tile.model_tiff_location))
        rmtree(os.path.dirname(results_path))


def raster_prediction(tif_path):
    with open('model_call_config.yml', 'r') as config:
        cfg = yaml.load(config, Loader=yaml.SafeLoader)
    model_api_cfg = cfg["model-api"]
    api_endpoint = "http://{host}:{port}/{endpoint}".format(
        host=model_api_cfg["host"],
        port=model_api_cfg["port"],
        endpoint=model_api_cfg["endpoint"]
    )
    data = {"image_path": tif_path}
    response = requests.post(url=api_endpoint, json=data)
    result = response.text
    datastore = json.loads(result)
    return datastore
