import os
import re
import sys

import imageio
import numpy as np
import yaml
from flask import Flask, abort, request, jsonify

sys.path.append('..')
from scripts.predict_raster import predict_raster, polygonize, save_polygons

app = Flask(__name__)


@app.route('/detect_clearcuts', methods=['POST'])
def detect_clearcuts():
    data = request.get_json()
    image_path = data['image_path']
    if len(image_path) == 0:
        abort(400)
    if '.' in image_path:
        filename = re.split(r'[./]', image_path)[-2]
    else:
        abort(400)
    predicted_filename = f'predicted_{filename}'
    channels, network, model_weights_path, save_path, threshold, input_size = load_config()
    result_directory_path = os.path.join(save_path, filename)
    os.makedirs(result_directory_path, exist_ok=True)

    raster_array, meta = predict_raster(
        image_path, channels,
        network, model_weights_path, input_size=input_size
    )
    save_raster(raster_array, result_directory_path, filename)

    polygons = polygonize(raster_array, meta, threshold)
    save_polygons(polygons, meta, result_directory_path, predicted_filename)
    return jsonify({"picture": os.path.join(result_directory_path, predicted_filename + '.png'),
                    "polygons": os.path.join(result_directory_path, predicted_filename + '.geojson')})


def save_raster(raster_array, save_path, filename):
    save_path = os.path.join(save_path, f'predicted_{filename}')
    imageio.imwrite(f'{save_path}.png', (raster_array * 255).astype(np.uint8))


def load_config():
    with open("config.yml", 'r') as config:
        cfg = yaml.load(config, Loader=yaml.SafeLoader)

    network = cfg['model']['network']
    channels = cfg['model']['channels']
    model_weights_path = cfg['model']['weights']
    save_path = cfg['prediction']['save_path']
    threshold = cfg['prediction']['threshold']
    input_size = cfg['prediction']['input_size']
    return channels, network, model_weights_path, save_path, threshold, input_size


if __name__ == '__main__':
    app.run(debug=True)
