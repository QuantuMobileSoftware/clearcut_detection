import json
import requests
import yaml


def raster_prediction(tif_path):
    with open('model_call_config.yml', 'r') as config:
        cfg = yaml.load(config, Loader=yaml.SafeLoader)
    model_api_cfg = cfg["model-api"]
    API_ENDPOINT = "http://{host}:{port}/{endpoint}".format(
        host=model_api_cfg["host"],
        port=model_api_cfg["port"],
        endpoint=model_api_cfg["endpoint"]
    )
    data = {"image_path": tif_path}
    r = requests.post(url=API_ENDPOINT, json=data)
    result = r.text
    datastore = json.loads(result)
    return datastore
