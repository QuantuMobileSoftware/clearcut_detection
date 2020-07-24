import os
import pandas as pd
import geopandas

from datetime import datetime
from shapely.ops import unary_union
from test.utils import DATE_CURRENT, DATE_PREVIOUS, TEST_POLYGONS
from test.settings import DATA_DIR


def save_polygons(polygons, crs, save_path, filename):
    if len(polygons) == 0:
        return
    polygons = geopandas.GeoDataFrame({'geometry': polygons}, crs = crs)
    polygons.to_file(os.path.join(save_path, f'{filename}.geojson'), driver='GeoJSON')
    return os.path.join(save_path, f'{filename}.geojson')


def prepare_testfile():
    test = geopandas.read_file(TEST_POLYGONS)
    test['img_date'] = pd.to_datetime(test['img_date'], format='%Y-%m-%d')
    test = test[(test['img_date'] > DATE_PREVIOUS) & (test['img_date'] <= DATE_CURRENT)]
    return test


def get_gt_polygons():
    test = prepare_testfile()
    clearcuts = unary_union(test['geometry'].buffer(1e-5)).buffer(-1e-5)
    return save_polygons(clearcuts, test.crs, DATA_DIR, 'test_clearcuts')
