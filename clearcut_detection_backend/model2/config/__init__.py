import os
import yaml

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

MEDIA_PATH = '/media/data'
