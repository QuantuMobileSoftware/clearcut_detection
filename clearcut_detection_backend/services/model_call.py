import os
import logging
from concurrent.futures import as_completed, ThreadPoolExecutor
from pathlib import Path
from distutils.util import strtobool
from django.conf import settings
from clearcuts.geojson_save import save_from_task
from clearcuts.models import TileInformation, RunUpdateTask, Tile
from services.prepare_tif import prepare_tiff
from clearcut_detection_backend import app

make_predict = strtobool(os.environ.get('MAKE_PREDICT', 'true'))

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
        with ThreadPoolExecutor(max_workers=int(settings.MAX_WORKERS/2)) as executor:
            future_list = []
            for tile_index in self.tile_index_distinct:
                tiles = TileInformation.objects.filter(tile_index__exact=tile_index,
                                                       source_tci_location__isnull=False,
                                                       source_b04_location__isnull=False,
                                                       source_b08_location__isnull=False,
                                                       source_b8a_location__isnull=False,
                                                       source_b11_location__isnull=False,
                                                       source_b12_location__isnull=False,
                                                       source_clouds_location__isnull=False)
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

        if make_predict:
            logger.info(f'send task_id: {task.id} to queue')
            result = model_add_task(task.id)
            res = result.get()
            try:
                save_from_task(res)
                return f'task_id: {res} done'
            except (ValueError, Exception):
                logger.error(f'cant do save_from_task({res})')
        else:
            logger.info(f'skip predict, save task_id: {task.id}')
            save_from_task(task.id)

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
