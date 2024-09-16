import os
import logging
from distutils.util import strtobool
from pathlib import Path
from config import models, threshold, input_size, PREDICTED_PATH
from utils import weights_exists_or_download
from predict_raster import predict_raster, polygonize, postprocessing, save_polygons
from services.run_predict_tasks_service import RunPredictTasks as RpT
from datetime import datetime

predict = strtobool(os.environ.get('PREDICT', 'true'))

logging.basicConfig(format='%(asctime)s %(message)s')

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
    filename = list_tif_path[3]
    predicted_filename = f'predicted_{filename}_{params["image_date_0"]}_{params["image_date_1"]}.geojson'
    result_directory_path = PREDICTED_PATH / filename
    result_directory_path.mkdir(parents=True, exist_ok=True)
    if not (result_directory_path / predicted_filename).is_file():

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

        if predict:
            raster_array, meta = predict_raster(
                params['path_img_1'],
                params['path_img_0'],
                channels,
                network,
                model_weights_path,
                input_size=input_size,
            )

            clearcuts = polygonize(raster_array > threshold, meta)
            cloud_files = [params['path_clouds_1'], params['path_clouds_0']]

            polygons = postprocessing(filename, cloud_files, clearcuts, meta['crs'])
            save_polygons(polygons, result_directory_path, predicted_filename)

        params['result'] = str(result_directory_path / predicted_filename)
        params['date_finished'] = str(datetime.now())

        RpT.update_task_by_id(session, task_id, params)
        return params['result']
    else:
        params['result'] = str(result_directory_path / predicted_filename)
        params['date_finished'] = str(datetime.now())
        RpT.update_task_by_id(session, task_id, params)
        logging.info(f'file {str(result_directory_path / predicted_filename)} exist. Skip')
        return str(result_directory_path / predicted_filename)
