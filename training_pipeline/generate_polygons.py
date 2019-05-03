from tqdm import tqdm
tqdm.monitor_interval = 0

from osgeo import ogr
import rasterio
import rasterio.mask
from shapely.geometry import Polygon, mapping
import geojson
from tqdm import tqdm

import os
import argparse
import json
import logging
import cv2
import numpy as np


class MaskGenerator:
    def __init__(self, raster_path, method='adaptive', mask_value=120):
        self.raster_array = None
        self.meta = None
        self.ndsm_part = None
        self.raster_path = raster_path
        self.method = method
        self.mask_value = mask_value

    def load_raster(self):
        with rasterio.open(self.raster_path, 'r') as src:
            # self.fwd = src.meta['transform']
            # self.rev = ~self.fwd
            data = src.read(1)
            self.raster_array = data
            self.meta = src.meta

    def process_ref(self, raster_type='masked'):
        array = self.raster_array.copy()
        array[array < 0] = 0
        array_ = np.uint8(array * 255)
        return array_

    def threshold_raster(self, raster_type='masked', eta=None):
        # if not eta:
        #     eta = self.ndsm_part
        ref_arr = self.process_ref(raster_type)
        # if 'ndsm' in self.raster_array and eta != 0:
        #     ndsm_arr = self.process_ndsm()
        #     # Add weighted array
        #     ref_arr = cv2.addWeighted(ndsm_arr, eta, ref_arr, (1 - eta), 0)

        ref_arr = cv2.medianBlur(ref_arr, 5)

        if self.method == 'binary':
            ret2, th2 = cv2.threshold(ref_arr, self.mask_value, 255, cv2.THRESH_BINARY)
            logging.info('{} thresholding was used, with threshold value {}'.format(self.method, self.mask_value))
        elif self.method == 'adaptive':
            th2 = cv2.adaptiveThreshold(ref_arr, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 99, -10)
            logging.info('{} thresholding was used'.format(self.method))
        else:
            ret2, th2 = cv2.threshold(ref_arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            logging.info('{} thresholding was used, with threshold value {}'.format(self.method, ret2))
        return th2

    def get_contours(self):
        th2 = self.threshold_raster()
        im2, contours, hierarchy = cv2.findContours(th2.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise TypeError("Contours wasn't found, Threshold- {} is too high".format(self.mask_value))
        return contours

    def polygonize(self, contours, transform=True):
        polygons = []
        for i in tqdm(range(len(contours))):
            c = contours[i]
            n_s = (c.shape[0], c.shape[2])
            if n_s[0] > 2:
                if transform:
                    polys = [tuple(i) * self.meta['transform'] for i in c.reshape(n_s)]
                else:
                    polys = [tuple(i) for i in c.reshape(n_s)]
                polygons.append(Polygon(polys))
        return polygons


def save_polys_as_shp(polys, name):
    # Now convert it to a shapefile with OGR
    driver = ogr.GetDriverByName('Esri Shapefile')
    ds = driver.CreateDataSource(name)
    layer = ds.CreateLayer('', None, ogr.wkbPolygon)
    # Add one attribute
    layer.CreateField(ogr.FieldDefn('id', ogr.OFTInteger))
    defn = layer.GetLayerDefn()

    # If there are multiple geometries, put the "for" loop here
    for i in range(len(polys)):
        # Create a new feature (attribute and geometry)
        feat = ogr.Feature(defn)
        feat.SetField('id', i)

        # Make a geometry, from Shapely object
        geom = ogr.CreateGeometryFromWkb(polys[i].wkb)
        feat.SetGeometry(geom)

        layer.CreateFeature(feat)
        # feat = geom = None  # destroy these

    # Save and close everything
    # ds = layer = feat = geom = None


def read_json(path):
    with open(path) as json_data:
        d = json.load(json_data)
    return d


def save_polys_as_json(polys, name, crs):
    # gc = GeometryCollection(polys)
    geoms = {}
    geoms['features'] = []
    geoms['crs'] = {'properties': {'name': 'urn:ogc:def:crs:EPSG::{}'.format(crs['init'].split(':')[-1]
)},
                    'type': 'name'}
    geoms['type'] = 'FeatureCollection'
    for i in range(len(polys)):

        geom_in_geojson = geojson.Feature(geometry=mapping(polys[i]), properties={})
        geoms['features'].append(geom_in_geojson)
    with open(name, 'w') as dst:
        # json.dumps()
        json.dump(geoms, dst)


def run(params):
    print(params.raster_path)
    d = read_json(params.raster_path)
    for raster, path in d.items():
        print(raster)
        out_path = "/home/user/projects/geo/dl/unet/polys_"
        mask_gen = MaskGenerator(raster_path=path)
        mask_gen.load_raster()
        contours = mask_gen.get_contours()
        polygons = mask_gen.polygonize(contours)
        out_path_j = os.path.join(out_path, raster + '.geojson')
        out_path_s = os.path.join(out_path, raster + '.shp')
        save_polys_as_json(polygons, out_path_j, mask_gen.meta['crs'])
        save_polys_as_shp(polygons, out_path_s)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--raster_path", help="Input raster path", type=str)
    # parser.add_argument("-o", "--out_path", help="output shp file", type=str)
    # parser.add_argument("-m", "--method", help="Thresholding method", default='adaptive', type=str)
    # parser.add_argument("-tv", "--thresh_val", help="Thresholding value", default=120, type=int)
    args = parser.parse_args()
    run(args)
