import json
import os
import logging
from concurrent.futures import as_completed, ThreadPoolExecutor
import requests
import yaml
from pathlib import Path
from django.conf import settings
from clearcuts.geojson_save import save
from clearcuts.models import TileInformation, RunUpdateTask, Tile
from test.prepare_tif import prepare_tiff
from services.configuration import area_tile_set_test

model_call_config = './model_call_config.yml'
logger = logging.getLogger('model_call')


class ModelCaller:
    """
    Class for asynchronous calling of model's API
    """
    def __init__(self):
        self.data_dir = settings.DATA_DIR
        self.tile_index_distinct = set(area_tile_set_test)
        logger.info(f'tile_index_distinct: {self.tile_index_distinct}')

    def start(self):
        if settings.PATH_TYPE == 'fs':
            path_type = 0
        else:
            logger.error(f'Unsupported file path in settings.PATH_TYPE: {settings.PATH_TYPE}')
            raise ValueError


        with ThreadPoolExecutor(max_workers=settings.MAX_WORKERS) as executor:
            future_list = []
            for tile_index in self.tile_index_distinct:
                tiles = [f'{tile_index}_0', f'{tile_index}_1']
                for tile_name in tiles:
                    logger.info(f'ThreadPoolExecutor submit {tile_index}, {tile_name}')
                    future = executor.submit(prepare_tiff, tile_name)
                    future_list.append(future)

        results = {}
        for future in as_completed(future_list):
            if future.result()[0]:
                self.remove_temp_files(future.result()[0], future.result()[1])
                if future.result()[2]:
                    tile_index = future.result()[2]
                    if tile_index not in results:
                        results[tile_index] = 1
                    else:
                        logger.info(f'start model_predict for {tile_index}')
                        self.model_predict(self.query.filter(tile_index__exact=tile_index))
                        del results[tile_index]

                        tile_list = self.query.filter(tile_index__exact=tile_index).order_by('tile_name')
                        path_img_0 = tile_list[0].model_tiff_location
                        path_img_1 = tile_list[1].model_tiff_location
                        image_date_0 = tile_list[0].tile_date
                        image_date_1 = tile_list[1].tile_date

                        tile = Tile.objects.get(tile_index=tile_index)
                        task = RunUpdateTask(tile_index=tile,
                                             path_type=path_type,
                                             path_img_0=path_img_0,
                                             path_img_1=path_img_1,
                                             image_date_0=image_date_0,
                                             image_date_1=image_date_1,
                                             )
                        task.save()




                    if len(results) > 0:
                        logger.error(f'results after model_predict not empty.\n\
                          results: {results}')

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
        logger.info(f'raster_prediction {tif_path}')
        results = raster_prediction(tif_path)
        logger.info(f'results:\n{results}')
        results_path = os.path.join(self.data_dir, results[0].get('polygons'))
        if os.path.exists(results_path):
            save(tile, results_path)

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
