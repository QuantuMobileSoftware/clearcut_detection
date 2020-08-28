import os
import logging
import time
from datetime import datetime, timedelta
from celery import group
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

    def run_all_from_prepared(self, tile_index):

        prepared = Prepared.objects.filter(success=1, tile_id__tile_index=tile_index).order_by('image_date')
        logger.info(f'len(prepared): {len(prepared)}')
        if len(prepared) < 2:
            logger.error(f'cant predict tile {tile_index}, len(prepared) < 2')
            return

        task_list = []
        for i in range(len(prepared)-1):
            path_img_0 = prepared[i].model_tiff_location
            path_img_1 = prepared[i + 1].model_tiff_location
            image_date_0 = prepared[i].image_date
            image_date_1 = prepared[i + 1].image_date

            path_clouds_0 = prepared[i].cloud_tiff_location
            path_clouds_1 = prepared[i + 1].cloud_tiff_location

            tile = Tile.objects.get(tile_index=tile_index)

            task = RunUpdateTask(tile=tile,
                                 path_type=path_type,
                                 path_img_0=path_img_0,
                                 path_img_1=path_img_1,
                                 image_date_0=image_date_0,
                                 image_date_1=image_date_1,
                                 path_clouds_0=path_clouds_0,
                                 path_clouds_1=path_clouds_1
                                 )
            task.save()

            logger.info(f'send task_id: {task.id} to queue')
            task_list.append(app.send_task(
                name='tasks.run_model_predict',
                queue='model_predict_queue',
                kwargs={'task_id': task.id},
                # task_track_started=True,
                ignore_result=False,
                # countdown=100,
                # timeout = 10000000,
                ))

        if make_predict:
            job = group(task_list)
            task_saved = False
            while len(job.tasks):
                time.sleep(2 * 10)
                logger.info(f'len(job.tasks): {len(job.tasks)}')
                cnt = 0
                for j in job.tasks:
                    logger.info(j)
                    logger.info(j.successful())
                    if j.successful():
                        logger.info(j.result)
                        task_id = j.result
                        job.tasks.pop(cnt)
                        try:
                            save_from_task(task_id)
                            task_saved = True
                        except (ValueError, Exception):
                            logger.error(f'cant do save_from_task({task_id})', exc_info=True)
                        if task_saved:
                            self.update_tile_dates(task_id)
                            task_saved = False
                        break
                    cnt += 1

    @staticmethod
    def update_tile_dates(task_id):
        task = RunUpdateTask.objects.get(id=task_id)
        tile = Tile.objects.get(id=task.tile.id)
        if tile.first_date is None or tile.first_date > task.image_date_0:
            tile.first_date = task.image_date_0

        if tile.last_date is None or tile.last_date < task.image_date_1:
            tile.last_date = task.image_date_1
        tile.save()
