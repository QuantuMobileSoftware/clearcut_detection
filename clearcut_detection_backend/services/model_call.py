import json
import os
import logging
from concurrent.futures import as_completed, ThreadPoolExecutor
import requests
import yaml
from pathlib import Path
from django.conf import settings
from clearcuts.geojson_save import save
from clearcuts.models import TileInformation
from services.prepare_tif import prepare_tiff
from services.email_on_error import emaile_on_service_error

model_call_config = './model_call_config.yml'
logger = logging.getLogger('model_call')


class ModelCaller:
    """
    Class for asynchronous calling of model's API
    """
    def __init__(self):
        self.data_dir = settings.DATA_DIR
        self.query = TileInformation.objects.filter(tile_name__isnull=False,
                                                    tile_index__isnull=False,
                                                    source_b04_location__isnull=False,
                                                    source_b08_location__isnull=False,
                                                    source_b8a_location__isnull=False,
                                                    source_b11_location__isnull=False,
                                                    source_b12_location__isnull=False,
                                                    source_tci_location__isnull=False,
                                                    source_clouds_location__isnull=False).order_by('tile_index')
       
        self.tile_index_distinct = list(TileInformation.objects.values_list(
            'tile_index', flat=True).distinct().order_by('tile_index')
                                        )

    def start(self):

        with ThreadPoolExecutor(max_workers=settings.MAX_WORKERS) as executor:
            future_list = []
            for tile in self.query:
                logger.info(f'ThreadPoolExecutor submit {tile.tile_index}, {tile.tile_name}')
                future = executor.submit(prepare_tiff, tile)
                future_list.append(future)

            for f in as_completed(future_list):
                self.remove_temp_files(f.result()[0], f.result()[1])

        for unique_tile_index in self.tile_index_distinct:
            logger.info(f'start model_predict for {unique_tile_index}')
            self.model_predict(self.query.filter(tile_index__exact=unique_tile_index))

    @staticmethod
    def remove_temp_files(path, tile_name):
        logger.info(f'Try remove temp files for {tile_name}')
        temp_files = Path(path).glob(f'{tile_name}*.tif')
        try:
            for file in temp_files:
                file.unlink()
            logger.info(f'temp files for {tile_name} were removed')
        except OSError:
            logger.error('Error\n\n', exc_info=True)

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
    except Exception as e:
        logger.error('Error\n\n', exc_info=True)
        subject = prepare_tiff.__qualname__
        emaile_on_service_error(subject, e)
