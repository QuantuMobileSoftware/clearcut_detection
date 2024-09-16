import os
import yaml
from pathlib import Path

with open('/model/predict_config.yml', 'r') as config:
    cfg = yaml.load(config, Loader=yaml.SafeLoader)

models = cfg['models']
save_path = cfg['prediction']['save_path']
threshold = cfg['prediction']['threshold']
input_size = cfg['prediction']['input_size']

POSTGRES_USER = os.environ.get('POSTGRES_USER', 'postgres')
POSTGRES_PASSWORD = os.environ.get('POSTGRES_PASSWORD', 'postgres')
POSTGRES_DB = os.environ.get('POSTGRES_DB', 'postgres')
DB_HOST = os.environ.get('DB_HOST', 'db_prod')

RABBITMQ_USER = os.environ.get('RABBITMQ_DEFAULT_USER', 'guest')
RABBITMQ_PASS = os.environ.get('RABBITMQ_DEFAULT_PASS', 'guest')
RABBITMQ_HOST = os.environ.get('RABBITMQ_HOST', 'rabbitmq_prod')
RABBITMQ_PORT_NUMBER = os.environ.get('RABBITMQ_NODE_PORT_NUMBER', 5672)
CUDA_VISIBLE_DEVICES = os.environ.get('CUDA_VISIBLE_DEVICES', '0')

SCOPES = ['https://www.googleapis.com/auth/drive.file']

LANDCOVER_POLYGONS_PATH = Path('/data/landcover')
PREDICTED_PATH = Path('/data/predicted')
LANDCOVER_FILENAME = 'S2A_OPER_GIP_TILPAR_MPC__20151209T095117_V20150622T000000_21000101T000000_B00.kml'
SENTINEL_TILES = LANDCOVER_POLYGONS_PATH / LANDCOVER_FILENAME
LANDCOVER_GEOJSON = LANDCOVER_POLYGONS_PATH / 'landcover_polygons.geojson'

CLOUDS_PROBABILITY_THRESHOLD = 15
NEAREST_POLYGONS_NUMBER = 10
DATES_FOR_TILE = 2
