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


# All of the loaded extensions. We don't want to load an extension twice.
_loaded_extensions = set()
_ws = None
_session = None
_task_id = None


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
    # ws_url = config.get('ws', 'url')
    # guard_header = config.get('ws', 'guard_header')
    # guard_key = config.get('ws', 'guard_key')
    # header = ["{}: {}".format(guard_header, guard_key)]

    params = RpT.get_task_by_id(_session, _task_id)
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

    path_array = []
    for model in models:
        predicted_filename = f'predicted_{model}_{filename}'

        channels = models[model]['channels']
        network = models[model]['network']
        try:
            model_weights_path = weights_exists_or_download(
                models[model]['weights'],
                os.environ.get('GOOGLE_DRIVE_FILE_ID'),
            )
        except (ValueError, Exception) as e:
            print(e)  # TODO

        raster_array, meta = predict_raster(
            str(image_path),
            channels,
            network,
            model_weights_path,
            input_size=input_size
        )

        save_raster(raster_array, result_directory_path, predicted_filename)

        claercuts = polygonize(raster_array > threshold, meta)
        print('11111111111111111111111')
        polygons = postprocessing(image_path, claercuts, meta['crs'])
        print('222222222222222222222222222')

        # print(polygons)
        # print(type(polygons))
        polygons_json = polygons.to_json()
        print(f'dddddddddddd: {polygons_json}')


        save_polygons(polygons, result_directory_path, predicted_filename)  # TODO
    
    return




    logger.info('simulation_start_date: {}'.format(params['simulation_start_date']))

    result = params['result']

    try:
        new_params = dict(status=1, simulation_start_date=str(datetime.now()))

        RpT.update_task_by_id(_session, _task_id, new_params)
        params.update(new_params)

    except RuntimeError:
        RpT.update_task_by_id(_session, task_id, {'status': 20})
        raise LogValueErrors(logger, log_msg=None)

    return


def save_raster(raster_array, save_path, filename):
    save_path = join(save_path, filename)
    imageio.imwrite(f'{save_path}.png', (np.abs(raster_array) * 255).astype(np.uint8))
