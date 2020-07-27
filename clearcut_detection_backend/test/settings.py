from pathlib import Path

MAXIMUM_CLOUD_PERCENTAGE_ALLOWED = 20.0
MAXIMUM_EMPTY_PIXEL_PERCENTAGE = 5.0
MAXIMUM_DATES_STORE_FOR_TILE = 2
MAX_WORKERS = 6

DATA_DIR = Path('./data/test')
DOWNLOADED_IMAGES_DIR = f'{DATA_DIR}/source_images/'
MODEL_TIFFS_DIR = f'{DATA_DIR}/model_tiffs'

METRIC_CONFIG = './test/metrics.ini'
TEST_CONFIG = './test/test_config.ini'
