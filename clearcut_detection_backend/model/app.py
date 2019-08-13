import os
import yaml
import imageio
import numpy as np
import re

from flask import Flask, abort, request, jsonify
from model.predict_raster import predict_raster, polygonize, save_polygons
from os.path import join

app = Flask(__name__)


@app.route('/raster_prediction', methods=['POST'])
def raster_prediction():
    data = request.get_json()
    image_path = data['image_path']
    if len(image_path) == 0:
        abort(400)
    if '.' in image_path:
        filename = re.split(r'[./]', image_path)[-2]
    else:
        abort(400)
    models, save_path, threshold, input_size = load_config()
    predicted_directory_name = 'predicted_' + filename
    result_directory_path = join(save_path, predicted_directory_name)
    os.makedirs(result_directory_path, exist_ok=True)
    path_array = []
    for model in models:
        predicted_filename = f'predicted_{model}_{filename}'

        channels = models[model]['channels']
        network = models[model]['network']
        model_weights_path = models[model]['weights']
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


def save_raster(raster_array, save_path, filename):
    save_path = join(save_path, filename)
    imageio.imwrite(f'{save_path}.png', (raster_array * 255).astype(np.uint8))


def load_config():
    with open('./predict_config.yml', 'r') as config:
        cfg = yaml.load(config, Loader=yaml.SafeLoader)

    models = cfg['models']
    save_path = cfg['prediction']['save_path']
    threshold = cfg['prediction']['threshold']
    input_size = cfg['prediction']['input_size']

    return models, save_path, threshold, input_size


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
