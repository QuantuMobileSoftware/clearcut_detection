import os
import yaml
import imageio
import numpy as np
import re
import traceback

from flask import Flask, abort, request, jsonify, make_response
from predict_raster import predict_raster, polygonize, save_polygons
from os.path import join

from utils import weights_exists_or_download

app = Flask(__name__)


@app.route('/raster_prediction', methods=['POST'])
def raster_prediction():
    data = request.get_json()
    image_path = data.get('image_path', '')
    if not image_path:
        abort(make_response(jsonify(message="Invalid request payload. It must contains image_path."), 400))
    if '.' in image_path:
        filename = re.split(r'[./]', image_path)[-2]
    else:
        abort(make_response(jsonify(message=f"Invalid image path ({image_path})"), 400))
    try:
        models, save_path, threshold, input_size = load_config()
        predicted_directory_name = 'predicted_' + filename
        result_directory_path = join(save_path, predicted_directory_name)
        os.makedirs(result_directory_path, exist_ok=True)
        path_array = []
        for model in models:
            predicted_filename = f'predicted_{model}_{filename}'

            channels = models[model]['channels']
            network = models[model]['network']
            model_weights_path = weights_exists_or_download(models[model]['weights'],
                                                            os.environ.get('GOOGLE_DRIVE_FILE_ID'))
            raster_array, meta = predict_raster(
                image_path, channels,
                network, model_weights_path, input_size=input_size
            )
            save_raster(raster_array, result_directory_path, predicted_filename)

            polygons = polygonize(raster_array, meta, threshold)
            save_polygons(polygons, meta, result_directory_path, predicted_filename)

            path_array.append({
                'picture': join(predicted_directory_name, predicted_filename + '.png'),
                'polygons': join(predicted_directory_name, predicted_filename + '.geojson')
            })
        return jsonify(path_array)
    except Exception as e:
        return make_response(jsonify(
            message=f'Model fail with next exception: '
                    f'\n\n{str(e)}\n\n {"".join(traceback.format_tb(e.__traceback__))}'),
            400
        )


def save_raster(raster_array, save_path, filename):
    save_path = join(save_path, filename)
    imageio.imwrite(f'{save_path}.png', (raster_array * 255).astype(np.uint8))


def load_config():
    with open('/model/predict_config.yml', 'r') as config:
        cfg = yaml.load(config, Loader=yaml.SafeLoader)

    models = cfg['models']
    save_path = cfg['prediction']['save_path']
    threshold = cfg['prediction']['threshold']
    input_size = cfg['prediction']['input_size']

    return models, save_path, threshold, input_size


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
