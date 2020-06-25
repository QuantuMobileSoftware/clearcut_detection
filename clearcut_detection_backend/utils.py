"""
Script helpers
"""
import os
import warnings
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
    landcover_path = './data/landcover'
    landcover_file = f'{landcover_path}/forest.tiff'
    os.system(f'./download_landcover.sh {landcover_path}')
    if not os.path.isfile(landcover_file):
        warnings.warn(f'{landcover_file} was not found.', UserWarning)
