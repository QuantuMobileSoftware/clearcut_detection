import os
import logging
from pathlib import Path
from distutils.util import strtobool
from django.conf import settings
from clearcut_detection_backend import app
from clearcuts.models import RunUpdateTask, TileInformation, Tile
from tiff_prepare.models import Prepared
from clearcuts.geojson_save import save_from_task

make_predict = strtobool(os.environ.get('MAKE_PREDICT', 'true'))
logger = logging.getLogger('create_update_task')

if settings.PATH_TYPE == 'fs':
    path_type = 0
else:
    logger.error(f'Unsupported file path in settings.PATH_TYPE: {settings.PATH_TYPE}')
    raise ValueError


class CreateUpdateTask:
    # def __init__(self, tile_index):
    #     self.tile_index = tile_index

    @staticmethod
    def run_all_from_prepared(tile_index):

        prepared = Prepared.objects.filter(success=1, tile_id__tile_index=tile_index).order_by('image_date')
        logger.info(f'len(prepared): {len(prepared)}')
        if len(prepared) < 2:
            logger.error(f'cant predict tile {tile_index}, len(prepared) < 2')
            return

        for i in range(len(prepared)-1):
            path_img_0 = prepared[i].model_tiff_location
            path_img_1 = prepared[i + 1].model_tiff_location
            image_date_0 = prepared[i].image_date
            image_date_1 = prepared[i + 1].image_date

            path_clouds_0 = prepared[i].cloud_tiff_location
            path_clouds_1 = prepared[i + 1].cloud_tiff_location

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
