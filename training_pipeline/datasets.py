import random

import cv2
import numpy as np
import pandas as pd

from keras.applications import imagenet_utils
from keras.preprocessing.image import load_img, img_to_array
from scipy.misc.pilutil import imread

from params import args
from sklearn.model_selection import train_test_split
import sklearn.utils
from random_transform_mask import ImageWithMaskFunction, pad_img, tiles_with_overlap, read_img_opencv, rgb2rgg, get_window
from random_transform_mask import pad as pad_folder
import os


def pad(image, padding_w, padding_h):
    batch_size, height, width, depth = image.shape
    # @TODO: Avoid creating new array
    new_image = np.zeros((batch_size, height + padding_h * 2, width + padding_w * 2, depth), dtype=image.dtype)
    new_image[:, padding_h:(height + padding_h), padding_w:(width + padding_w)] = image
    # @TODO: Fill padded zones
    # new_image[:, :padding_w] = image[:, :padding_w]
    # new_image[:padding_h, :] = image[:padding_h, :]
    # new_image[-padding_h:, :] = image[-padding_h:, :]

    return new_image

def min_max(X):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (1 - 0) + 0
    return X_scaled

def generate_images(img):
    output = np.zeros((img.shape[0], img.shape[1], 3))
    exr = 1.4 * img[:, :, 0] - img[:, :, 1]
    exr[exr < 0] = 0
    exr = np.uint8(exr)
    output[:, :, 0] = img[:, :, 0]/255
    output[:, :, 1] = img[:, :, 1]/255
    output[:, :, 2] = exr/255
    # output[:, :, 3] = min_max(cv2.Canny(exr, 30, 60))
    # output[:, :, 4] = min_max(cv2.Laplacian(exr, cv2.CV_64F))
    # output[:, :, 5] = min_max(cv2.Sobel(exr, cv2.CV_64F, 1, 0, ksize=7))
    # output[:, :, 6] = min_max(cv2.Sobel(exr, cv2.CV_64F, 0, 1, ksize=7))
    # chan_list = [min_max(chan) for chan in [edges, laplacian, sobelx, sobely]]
    return output

def unpad(image, padding_w):
    return image[:, :, padding_w:(image.shape[1] - padding_w), :]


def generate_filenames(car_ids):
    return ['{}_{}.jpg'.format(id, str(angle + 1).zfill(2)) for angle in range(16) for id in car_ids]

def bootstrapped_split(car_ids, seed=args.seed):
    """
    # Arguments
        metadata: metadata.csv provided by Carvana (should include
        `train` column).

    # Returns
        A tuple (train_ids, test_ids)
    """
    all_ids = pd.Series(car_ids)
    train_ids, valid_ids = train_test_split(car_ids, test_size=args.test_size_float,
                                                     random_state=seed)

    np.random.seed(seed)
    bootstrapped_idx = np.random.random_integers(0, len(train_ids))
    bootstrapped_train_ids = train_ids[bootstrapped_idx]

    return generate_filenames(bootstrapped_train_ids.values), generate_filenames(valid_ids)


def build_batch_generator(filenames, img_man_dir=None, batch_size=None,
                          shuffle=False, transformations=None,
                          out_size=None, crop_size=None, mask_dir=None, aug=False, r_type='rgg'):
    mask_function = ImageWithMaskFunction(out_size=out_size, crop_size=crop_size, mask_dir=mask_dir)

    while True:
        # @TODO: Should we fixate the seed here?
        if shuffle:
            filenames = sklearn.utils.shuffle(filenames)

        for start in range(0, len(filenames), batch_size):
            batch_x = []
            weights = []
            end = min(start + batch_size, len(filenames))
            train_batch = filenames[start:end]

            for ind, filename in train_batch.iterrows():
                ##TODO fix code to download image from disk
                img_path = os.path.join(img_man_dir, filename['folder'], '{}'.format(r_type))
                    # mask_path = os.path.join(img_man_dir)
                img = img_to_array(
                    load_img(os.path.join(img_path, filename['name'].replace('nrg', r_type).replace('rgg', r_type).replace('.png', '.jpg')), grayscale=False))
                batch_x.append(img)
                weights.append(filename['weight'])
            batch_x = np.array(batch_x, np.float32)
            batch_x, masks = mask_function.mask_pred(batch_x, train_batch, range(batch_size), img_man_dir, aug, r_type)
            weights = np.array(weights)
            if crop_size is None:
                # @TODO: Remove hardcoded padding
                batch_x, masks = pad(batch_x, 1, 0), pad(masks, 1, 0)
            if args.edges:
                yield batch_x, masks, weights
            else:
                yield imagenet_utils.preprocess_input(batch_x, mode=args.preprocessing_function), masks, weights

def build_batch_generator_predict_folder(filenames, overlap=0.5, batch_size=None):
    while True:
        # @TODO: Should we fixate the seed here?
        #x = []

        img_sizes = []
        for j in range(batch_size):
            if batch_size + j < len(filenames):
                img_path = os.path.join(args.test_data_dir, filenames[batch_size + j])
                img = read_img_opencv(img_path)
                tile_size = get_window(img, attitude=40, window_meters=30)
                img_size = img.shape

                img = rgb2rgg(img)
                img_sizes.append(img_size)
                tls, rects = tiles_with_overlap(img, tile_size, overlap)
                tile_sizes = [tile.shape for tile in tls]
                padded_tiles = []
                pads = []
                for tile in tls:
                    if tile.shape[0] != tile_size or tile.shape[1] != tile_size:
                        padded_tile, pad_ = pad_folder(tile, tile_size)
                        padded_tiles.append(padded_tile)
                        pads.append(pad_)
                    else:
                        padded_tiles.append(tile)
                        pads.append((0, 0, 0, 0))
                # tls = [img_to_array(cv2.resize(tile, (args.input_height, args.input_width))) for tile in tls]
                tls = [img_to_array(cv2.resize(tile, (args.input_height, args.input_width))) for tile in
                       padded_tiles]
                # tile_sizes = [tile.shape for tile in tls]
                # x.append(tls)
            x = np.array(tls)
            yield imagenet_utils.preprocess_input(x, mode=args.preprocessing_function)# , masks, weights


def get_edges(image):
    img = np.uint8(image[:,:,0])
    out = np.zeros((args.img_height, args.img_width, 5))
    laplacian = min_max(cv2.Laplacian(img, cv2.CV_64F))
    sobelx = min_max(cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5))
    sobely = min_max(cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5))
    edges = min_max(cv2.Canny(img, 30, 200))
    for n, c in zip(range(5), [image[:, :, 0]/255, laplacian, sobelx, sobely, edges]):
        out[:, :, n] = c
    return out


def min_max(X):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (1 - 0) + 0
    return X_scaled

