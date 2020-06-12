"""
Script helpers
"""
import os
from enum import Enum


def path_exists_or_create(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


class Bands(Enum):
    TCI = 'TCI'
    B04 = 'B04'
    B08 = 'B08'
    B8A = 'B8A'
    B11 = 'B11'
    B12 = 'B12'


def get_landcover():
	if not os.path.isfile('./data/landcover/forest.tiff'):
		os.system('apt update && apt-get install -y wget')
		os.system('wget https://s3-eu-west-1.amazonaws.com/vito.landcover.global/2015/E020N60_ProbaV_LC100_epoch2015_global_v2.0.2_products_EPSG-4326.zip -O landcover.zip')
		os.system('mv landcover.zip ./data/')
		os.system('cd data && unzip landcover.zip')
		path = path_exists_or_create('./data/landcover')
		os.system(f'mv ./data/E020N60_ProbaV_LC100_epoch2015_global_v2.0.2_forest-type-layer_EPSG-4326.tif {path}/forest.tiff')
		os.system('rm ./data/*.tif ./data/landcover.zip')
