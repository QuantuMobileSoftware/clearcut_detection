import os
# from os.path import join
# import imageio
# import numpy as np
from pathlib import Path
from config import models, threshold, input_size, save_path
from utils import weights_exists_or_download
from predict_raster import predict_raster, polygonize, postprocessing, save_polygons
from services.run_predict_tasks_service import RunPredictTasks as RpT
from datetime import datetime, timezone


def run_predict(session, task_id):
    """
    TODO
    """
    model_weights_path = None

    params = RpT.get_task_by_id(session, task_id)
    params['date_started'] = str(datetime.now())
    RpT.update_task_by_id(session, task_id, params)
    image_path = Path(params['path_img_0'])
    list_tif_path = list(image_path.parts)
    filename = list_tif_path[2]
    predicted_filename = f'predicted_{filename}.geojson'
    list_tif_path = list_tif_path[:1]
    list_tif_path.append('predicted')
    list_tif_path.append(filename)
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

    clearcuts = polygonize(raster_array > threshold, meta)
    cloud_files = [params['path_clouds_0'], params['path_clouds_1']]

    polygons = postprocessing(filename, cloud_files, clearcuts, meta['crs'])  # TODO
    save_polygons(polygons, result_directory_path, predicted_filename)

    params['result'] = str(result_directory_path / predicted_filename)
    params['date_finished'] = str(datetime.now())

    RpT.update_task_by_id(session, task_id, params)
    RpT.update_tileinformation(session, params['tile_index_id'])

    return params['tile_index_id']
