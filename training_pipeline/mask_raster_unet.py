import sys
from tqdm import tqdm
tqdm.monitor_interval = 0
import shutil


import json
from osgeo import ogr
import rasterio.mask
from keras.applications import imagenet_utils
from shapely.geometry import Polygon
import logging
import os
import cv2
from keras.preprocessing.image import flip_axis

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import keras.backend as K

from models import make_model

from params import args
from random_transform_mask import pad_size, unpad

logging.basicConfig(stream=sys.stdout, format='%(asctime)s %(message)s', datefmt=' %I:%M:%S ', level="INFO")

import rasterio
import numpy as np

def do_tta(x, tta_type):
    if tta_type == 'hflip':
        # batch, img_col = 2
        return flip_axis(x, 2)
    elif tta_type == 'vflip':
        # batch, img_col = 2
        return flip_axis(x, 1)
    else:
        return x


def undo_tta(pred, tta_type):
    if tta_type == 'hflip':
        # batch, img_col = 2
        return flip_axis(pred, 2)
    elif tta_type == 'vflip':
        # batch, img_col = 2
        return flip_axis(pred, 1)
    else:
        return pred

class Reader:
    def __init__(self, raster_list: dict):
        self.raster_array = None
        self.meta = None
        self.raster_list = raster_list

    def load_stack(self):
        self.raster_array = {}
        self.meta = {}
        for r_type, path in self.raster_list.items():
            with rasterio.open(path, 'r') as src:
                self.raster_array[r_type] = src.read()
                self.meta[r_type] = src.meta

    def create_nrg(self, int_float=False):
        green_path = self.raster_list['green']
        path = self.raster_list['green'].split('/')
        path[-1] = path[-1].replace('green', 'nrg')
        self.raster_list['nrg'] = "/".join(path)
        pathes = self.raster_list
        self.meta = {}

        with rasterio.open(green_path, 'r') as src:
            meta = src.meta
            meta['count'] = 3
            gr = src.read()
            if 2 < gr.max() <= 255:
                int_float = True
            if int_float:
                meta['dtype'] = 'uint8'
                meta['nodata'] = 0
            self.meta['nrg'] = meta
            # print(save_path)
        with rasterio.open(self.raster_list['nrg'], 'w', **meta) as dst:
            for i, raster in enumerate(['nir', 'red', 'green']):
                with rasterio.open(pathes[raster], 'r') as src:
                    if int_float:
                        arr = src.read(1).astype('uint8')
                    else:
                        arr = src.read(1)
                    if arr.max() > 256:
                        #raise AssertionError("Raster is not scaled properly max_value- {}".format(arr.max()))
                        logging.error("Raster is not scaled properly max_value- {}".format(arr.max()))
                        arr = self.min_max(arr, X_min=0, X_max=65536)
                    elif arr.max() < 0.1:
                        logging.error("Raster is not scaled properly max_value- {}".format(arr.max()))
                        arr[arr < 0] = 0
                        arr = self.min_max(arr)
                        #raise AssertionError("Raster is not scaled properly max_value- {}".format(arr.max()))
                    dst.write(arr, i+1)
        logging.info('nrg created')

    def create_rgg(self, int_float=False):
        if not ('rgbn' in self.raster_list or 'rgb' in self.raster_list):
            path = self.raster_list['green'].split('/')
            path[-1] = path[-1].replace('green', 'rgg')
            self.raster_list['rgg'] = "/".join(path)
            self.meta = {}
            pathes = self.raster_list
            with rasterio.open(self.raster_list['green'], 'r') as src:
                meta = src.meta
                meta['count'] = 3
                self.meta['rgg'] = meta
                gr = src.read()
                if 1 < gr.max() <= 255:
                    int_float = True
                if int_float:
                    meta['dtype'] = 'uint8'
                    meta['nodata'] = 0
                # print(save_path)
            with rasterio.open(self.raster_list['rgg'], 'w', **meta) as dst:
                for i, raster in enumerate(['red', 'green', 'green']):
                    with rasterio.open(pathes[raster], 'r') as src:
                        if int_float:
                            arr = src.read(1).astype('uint8')
                        else:
                            arr = src.read(1)
                        if arr.max() > 1 or arr.max() < 0.1:
                            logging.error("Raster is not scaled properly max_value- {}".format(arr.max()))

                        dst.write(arr, i+1)
        else:
            if 'rgb' in self.raster_list:
                path = self.raster_list['rgb'].split('/')
                rp = self.raster_list['rgb']
            else:
                path = self.raster_list['rgbn'].split('/')
                rp = self.raster_list['rgbn']
            path[-1] = path[-1].split('.')[0] + '_rgg.tif'
            self.raster_list['rgg'] = "/".join(path)
            self.meta = {}
            with rasterio.open(rp, 'r') as src:
                meta = src.meta
                meta['count'] = 3
                self.meta['rgg'] = meta
                # print(save_path)
                with rasterio.open(self.raster_list['rgg'], 'w', **meta) as dst:
                    for i in range(1, meta['count'] + 1):
                        if i == 3:
                            dst.write(src.read(2), i)
                        else:
                            dst.write(src.read(i), i)
        print('rgg created')

    def save_raster(self, raster_type, save_path=None):
        if not save_path:
            save_path = self.raster_list[raster_type]
        with rasterio.open(save_path, 'w', **self.meta[raster_type]) as dst:
            for i in range(1, self.meta[raster_type]['count'] + 1):
                src_array = self.raster_array[raster_type][i - 1]
                dst.write(src_array, i)

    def get_rgg(self):
        return self.raster_array['rgg'], self.meta['rgg']

    def get_nrg(self):
        return self.raster_array['nrg'], self.meta['nrg']

    @staticmethod
    def min_max(X, X_min=None, X_max=None):
        if X_min:
            X_std = (X - X_min) / (X_max - X_min)
            X_scaled = X_std * (1 - 0) + 0
        else:
            X_std = (X - X.min()) / (X.max() - X.min())
            X_scaled = X_std * (1 - 0) + 0
        return X_scaled


class SegmentatorNN:
    def __init__(self, inp_list, thresh, r_type):
        # self.raster_path = raster_path
        # self.raster_type = raster_type
        self.r_type = r_type
        inp = read_json(inp_list)
        self.thresh = thresh
        self.reader = Reader(inp)
        if 'rgb' in inp.keys():
            if 'rgg' not in inp.keys():
                self.reader.create_rgg()
                self.r_type = 'rgg'
        else:
            if self.r_type not in inp.keys():
                if self.r_type == 'rgg':
                    self.reader.create_rgg()
                else:
                    self.reader.create_nrg()

    def mask_tiles(self, save_mask=True, window_size=30):
        # batch_size = 1
        # config = tf.ConfigProto()
        # config.gpu_options.per_process_gpu_memory_fraction = 1
        # set_session(tf.Session(config=config))
        model = make_model((None, None, 3))
        model.load_weights(args.weights)
        max_values = [1, 1, 1]
        min_values = [0, 0, 0]
        with rasterio.open(self.reader.raster_list[self.r_type], 'r') as dataset:
            meta = dataset.meta
            raster_array = np.zeros((dataset.meta['height'], dataset.meta['width']), np.float32)
            xs = dataset.bounds.left
            window_size_meters = window_size
            window_size_pixels = window_size / (dataset.res[0])
            cnt = 0
            pbar = tqdm()
            while xs < dataset.bounds.right:
                ys = dataset.bounds.bottom
                while ys < dataset.bounds.top:
                    row, col = dataset.index(xs, ys)
                    pbar.set_postfix(Row='{}'.format(row), Col='{}'.format(col))
                    step_row = row - int(window_size_pixels)
                    step_col = col + int(window_size_pixels)
                    res = dataset.read(window=((max(0, step_row), row),
                                               (col, step_col)))
                    rect = [[max(0, step_row), row], [col, step_col]]
                    # if res.max() > 0:
                    #     print('hi')
                    if res.dtype == 'float32':
                        if res.max() > 1 or res.max() < 0.02:
                            res[res < 0] = 0
                            if 'rgb' not in self.reader.raster_list.keys() and 'rgbn' not in self.reader.raster_list.keys():
                                res = self.min_max(res, min=min_values, max=max_values)
                        if 'rgb' not in self.reader.raster_list.keys() and 'rgbn' not in self.reader.raster_list.keys():
                            if res.max() < 2:
                                res = self.process_float(res)
                        res = res.astype(np.uint8)
                    img_size = tuple([res.shape[2], res.shape[1]])
                    cv_res = raster_to_img(res, self.reader.raster_list, self.r_type)
                    cv_res = cv2.resize(cv_res, (args.input_width, args.input_width))
                    cv_res = np.expand_dims(cv_res, axis=0)
                    x = imagenet_utils.preprocess_input(cv_res, mode=args.preprocessing_function)
                    if args.pred_tta:
                        x_tta = do_tta(x, "hflip")
                        batch_x = np.concatenate((x, x_tta), axis=0)
                    else:
                        batch_x = x
                    # cv_res, pads = pad_size(cv_res)
                    #cv_res = np.expand_dims(cv_res, axis=0)
                    #x = imagenet_utils.preprocess_input(cv_res, mode=args.preprocessing_function)
                    preds = model.predict_on_batch(batch_x)
                    if args.pred_tta:
                        preds_tta = undo_tta(preds[1:2], "hflip")
                        pred = (preds[:1] + preds_tta) / 2
                    else:
                        pred = preds
                    pred = cv2.resize(pred[0], img_size)
                    # pred = unpad(pred[0], pads)
                    stack_arr = np.dstack([pred, raster_array[rect[0][0]:rect[0][1], rect[1][0]:rect[1][1]]])

                    raster_array[rect[0][0]:rect[0][1], rect[1][0]:rect[1][1]] = np.amax(stack_arr, axis=2)
                    # raster_array[rect[0][0]:rect[0][1], rect[1][0]:rect[1][1]] = np.mean(stack_arr, axis=2)

                    pbar.update(1)

                    cnt += 1
                    ys = ys + 0.5 * window_size_meters

                xs = xs + 0.5 * window_size_meters

        # Save raster
        bin_meta = meta.copy()
        bin_meta['count'] = 1
        bin_meta['dtype'] = 'float32'
        bin_meta['nodata'] = -10000
        bin_raster_path = self.reader.raster_list[self.r_type].replace(self.r_type, '{}_bin'.format(self.r_type))
        raster_name = bin_raster_path.split("/")[-1]
        raster_dir = os.path.dirname(bin_raster_path)
        save_raster_dir = os.path.join(raster_dir, "v5_tta_full")
        os.makedirs(save_raster_dir, exist_ok=True)
        save_raster_path = os.path.join(save_raster_dir, raster_name)
        save_single_raster(np.expand_dims(raster_array, axis=0), bin_meta, save_raster_path)
        raster_array = raster_array > self.thresh
        raster_array = (raster_array * 255).astype(np.uint8)
        if save_mask:
            cv2.imwrite(save_raster_path.replace('.tif', '_mask.jpg'), raster_array)
        im2, contours, hierarchy = cv2.findContours(raster_array.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        polygons = self.polygonize(contours, bin_meta)

        poly_path = os.path.dirname(self.reader.raster_list[self.r_type])
        poly_path = os.path.join(poly_path, 'polygons')
        if os.path.isdir(poly_path):
            shutil.rmtree(poly_path, ignore_errors=True)
        os.makedirs(poly_path, exist_ok=True)
        poly_path = os.path.join(poly_path, self.reader.raster_list[self.r_type].split('/')[-1].replace('.tif', '.shp'))
        try:
            if len(polygons) != 0:
                save_polys_as_shp(polygons, poly_path)
            else:
                print('no_polygons detected')
        except:
            print('done before')
        del model
        K.clear_session()
        return raster_array
        # raster_array = raster_array.astype(np.uint8)
        # self.save_raster_test(path, self.meta['masked'], raster_array)
        # raster_array = None

    def polygonize(self, contours, meta, transform=True):
        polygons = []
        for i in tqdm(range(len(contours))):
            c = contours[i]
            n_s = (c.shape[0], c.shape[2])
            if n_s[0] > 2:
                if transform:
                    polys = [tuple(i) * meta['transform'] for i in c.reshape(n_s)]
                else:
                    polys = [tuple(i) for i in c.reshape(n_s)]
                polygons.append(Polygon(polys))
        return polygons

    @staticmethod
    def process_float(array):
        array = array.copy()
        array[array < 0] = 0
        array_ = np.uint8(array * 255)
        return array_

    @staticmethod
    def min_max(X, min, max):
        X_scaled = np.zeros(X.shape)
        for i in range(X.shape[0]):
            X_std = (X[i] - min[i]) / (max[i] - min[i])
            X_scaled[i] = X_std * (1 - 0) + 0

        return X_scaled


def save_single_raster(raster_array, meta, save_path):
    with rasterio.open(save_path, 'w', **meta) as dst:
        for i in range(1, meta['count'] + 1):
            src_array = raster_array[i - 1]
            dst.write(src_array, i)


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
    #logging.info(path)
    with open(path, 'r') as json_data:
        d = json.load(json_data)
    return d


def raster_to_img(raster, raster_list, r_type):
    if raster.shape[0] > 1:
        cv_res = np.zeros((raster.shape[1], raster.shape[2], raster.shape[0]))
        if 'rgb' in raster_list.keys() and np.mean(cv_res) > 90:
            cv_res[:, :, 0] = raster[0] / 5.87
            cv_res[:, :, 1] = raster[1] / 5.95
            cv_res[:, :, 2] = raster[2] / 5.95
        elif 'rgbn' in raster_list.keys() and r_type == 'rgg':
            cv_res[:, :, 0] = raster[0] / 5.76
            cv_res[:, :, 1] = raster[1] / 5.08
            cv_res[:, :, 2] = raster[2] / 5.08
        elif 'rgbn' in raster_list.keys() and r_type == 'nrg':
            cv_res[:, :, 0] = raster[0] / 2.27
            cv_res[:, :, 1] = raster[1] / 5.76
            cv_res[:, :, 2] = raster[2] / 5.08
        else:
            cv_res[:, :, 0] = raster[0]
            cv_res[:, :, 1] = raster[1]
            cv_res[:, :, 2] = raster[2]
    else:
        cv_res = raster[0]
    return cv_res

def run():
    d = read_json(args.inp_list)
    thresh = args.threshold
    r_type = args.r_type
    for grove, inp in d.items():
        segmentator = SegmentatorNN(inp, thresh, r_type)
        segmentator.mask_tiles()


if __name__ == '__main__':
    # predict()
    run()