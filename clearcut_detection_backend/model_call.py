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
                                                    tile_index__isnull=False,
                                                    source_b04_location__isnull=False,
                                                    source_b08_location__isnull=False,
                                                    source_b8a_location__isnull=False,
                                                    source_b11_location__isnull=False,
                                                    source_b12_location__isnull=False,
                                                    source_tci_location__isnull=False,
                                                    source_clouds_location__isnull=False)
       
        self.distinct = TileInformation.objects.values('tile_index').distinct()

        self.data_dir = 'data'
        self.executor = ThreadPoolExecutor(max_workers=10)

    def select_by_index(self, tile_index):
        return TileInformation.objects.filter(tile_index__exact=tile_index)

    def start(self):
        for tile in self.query[:2]:
            self.executor.submit(self.preprocess, tile)
        for unique_tile_index in list(TileInformation.objects.values('tile_index').distinct()):
            tile_index = unique_tile_index['tile_index']
            self.model_predict(self.select_by_index(tile_index))

    def preprocess(self, tile):
        """
        Converting jp2file to tiff
        """
        prepare_tiff(tile)
    
    def model_predict(self, tile):
        """
        Sending unique tile_index to model and saving results to db
        """
        src_tile = tile.first()
        tif_path = src_tile.model_tiff_location.split('/')[:-2]
        tif_path = os.path.join(*tif_path)
        results = raster_prediction(tif_path)
        
        results_path = os.path.join(self.data_dir, results[0].get('polygons'))
        if os.path.exists(results_path):
            save(tile, results_path, forest=1)

        results_path = os.path.join(self.data_dir, results[0].get('polygons_not_forest'))
        if os.path.exists(results_path):
            save(tile, results_path, forest=0)
        
        #rmtree(os.path.dirname(tile.tile_index))
        #rmtree(os.path.dirname(results_path))

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
