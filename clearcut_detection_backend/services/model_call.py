import json
import os
import logging
from celery import group
from concurrent.futures import as_completed, ThreadPoolExecutor
import requests
import yaml
from pathlib import Path
from django.conf import settings
from clearcuts.geojson_save import save, save_from_task
from clearcuts.models import TileInformation, RunUpdateTask, Tile
from services.prepare_tif import prepare_tiff
from clearcut_detection_backend import app

model_call_config = './model_call_config.yml'
logger = logging.getLogger('model_call')

if settings.PATH_TYPE == 'fs':
    path_type = 0
else:
    logger.error(f'Unsupported file path in settings.PATH_TYPE: {settings.PATH_TYPE}')
    raise ValueError


class ModelCaller:
    """
    Class for asynchronous  calling of model's API
    """
    def __init__(self):
        self.data_dir = settings.DATA_DIR
        self.tile_index_distinct = Tile.objects.filter(is_tracked=1)
        logger.info(f'tile_index_distinct: {self.tile_index_distinct}')

    def start(self):
        with ThreadPoolExecutor(max_workers=settings.MAX_WORKERS) as executor:
            future_list = []
            for tile_index in self.tile_index_distinct:
                tiles = TileInformation.objects.filter(tile_index__exact=tile_index)
                if len(tiles) < 2:
                    logger.error(f'tile_index: {tile_index}, len(tiles) < 2')
                else:
                    for tile in tiles:
                        logger.info(f'ThreadPoolExecutor submit {tile.tile_index}, {tile.tile_name}')
                        future = executor.submit(prepare_tiff, tile)
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
                        del results[tile_index]

                        self.predict_and_save_from_result(tile_index)

                        if len(results) > 0:
                            logger.error(f'results after model_predict not empty.\n\
                              results: {results}')

    @staticmethod
    def predict_and_save_from_result(tile_index):
        tile_list = TileInformation.objects.filter(
            tile_index__tile_index=tile_index,
            is_prepared=1
        ).order_by('tile_name')
        if len(tile_list) < 2:
            logger.error(f'cant predict tile {tile_index}, len(tiles) < 2')
            return
        path_img_0 = tile_list[0].model_tiff_location
        path_img_1 = tile_list[1].model_tiff_location
        image_date_0 = tile_list[0].tile_date
        image_date_1 = tile_list[1].tile_date

        path_clouds_0 = str(Path(path_img_0).parent / 'clouds.tiff')
        path_clouds_1 = str(Path(path_img_1).parent / 'clouds.tiff')

        tile = Tile.objects.get(tile_index=tile_index)  # TODO

        task = RunUpdateTask(tile_index=tile,
                             path_type=path_type,
                             path_img_0=path_img_0,
                             path_img_1=path_img_1,
                             image_date_0=image_date_0,
                             image_date_1=image_date_1,
                             path_clouds_0=path_clouds_0,
                             path_clouds_1=path_clouds_1
                             )
        task.save()

        result = model_add_task(task.id)
        res = result.get()
        try:
            save_from_task(res)
            return f'task_id: {res} done'
        except Exception:
            logger.error(f'cant do save_from_task({res})')

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
        host=os.environ.get('MODEL_HOST', 'model'),
        port=os.environ.get('MODEL_PORT', 5000),
        endpoint=os.environ.get('MODEL_ENDPOINT', 'raster_prediction')
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


def model_add_task(task_id):
    """
    Add run update task task in to 'run_update_task' queue
    :param task_id:
    :return:
    """
    result = app.send_task(
        name='tasks.run_model_predict',
        queue='model_predict_queue',
        kwargs={'task_id': task_id},
        task_track_started=True,
        )
    return result
