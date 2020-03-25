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
