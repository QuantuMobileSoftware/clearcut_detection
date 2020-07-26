import os
from os.path import join
import yaml
import imageio
import numpy as np
# import re
# import json
# import warnings
#
# import pickle
# from datetime import datetime, timezone
#
# from config import config
# from services.run_algorithm_tasks_logs_services import logger
# from services.errors_logs_services import LogValueErrors
from pathlib import Path
from config import models, save_path, threshold, input_size
from utils import weights_exists_or_download
from predict_raster import predict_raster, polygonize, save_polygons, postprocessing
from services.run_predict_tasks_service import RunPredictTasks as RpT
from datetime import datetime, timezone


_session = None
_task_id = None
need_predict = True


def run_predict(session, task_id):
    """
    Run algorithm with params
    """
    global _ws
    global _session
    global _task_id
    _session = session
    _task_id = task_id
    model_weights_path = None

    params = RpT.get_task_by_id(_session, _task_id)
    params['date_started'] = str(datetime.now())
    RpT.update_task_by_id(_session, _task_id, params)

    image_path = Path(params['path_img_0']).parent.parent
    list_tif_path = list(image_path.parts)
    filename = list_tif_path[-1]
    print(filename)
    predicted_directory_name = f'predicted_{filename}'

    print(list_tif_path)
    list_tif_path[-1] = f'predicted_{filename}'
    print(list_tif_path)
    result_directory_path = Path(*list_tif_path)
    print(result_directory_path)
    result_directory_path.mkdir(parents=True, exist_ok=True)

    channels = models['deforestration_detection']['channels']
    network = models['deforestration_detection']['network']
    weights = models['deforestration_detection']['weights']

    try:
        model_weights_path = weights_exists_or_download(
            weights,
            os.environ.get('GOOGLE_DRIVE_FILE_ID'),
        )
    except (ValueError, Exception) as e:
        print(e)  # TODO

    raster_array, meta = predict_raster(
        params['path_img_0'],
        params['path_img_1'],
        channels,
        network,
        model_weights_path,
        input_size=input_size,
    )

    # save_raster(raster_array, result_directory_path, predicted_filename)

    clearcut = polygonize(raster_array > threshold, meta)

    polygons = postprocessing(image_path, clearcut, meta['crs'])  # TODO
    polygons_json = polygons.to_json()

    params['result'] = polygons_json
    params['date_finished'] = str(datetime.now())

    RpT.update_task_by_id(_session, _task_id, params)
    RpT.update_tileinformation(_session, params['tile_index_id'])

    # save_polygons(polygons, result_directory_path, predicted_filename)  # TODO

    # logger.info('simulation_start_date: {}'.format(params['simulation_start_date']))
    return


def save_raster(raster_array, save_path, filename):
    save_path = join(save_path, filename)
    imageio.imwrite(f'{save_path}.png', (np.abs(raster_array) * 255).astype(np.uint8))
