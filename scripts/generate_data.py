import os
import imageio
import numpy as np
import pandas as pd
import geopandas as gp
from sklearn.model_selection import StratifiedShuffleSplit


def get_data_pathes(
    datasets_path, images_path_name='images',
    masks_path_name='masks', instances_path_name='instance_masks'
    ):
    
    datasets = list(os.walk(datasets_path))[0][1]
    data_pathes = []
    for dataset in datasets:
        data_pathes.append((
            os.path.join(datasets_path, dataset, images_path_name),
            os.path.join(datasets_path, dataset, masks_path_name),
            os.path.join(datasets_path, dataset, instances_path_name)))
    
    return data_pathes


def get_instances(instances_path):
    return list(os.walk(instances_path))[0][1]


def image2mask(image_path, image_type):
    return imageio.imread('{}.{}'.format(image_path, image_type))


def get_data(
    images_path, masks_path, instances,
    img_type='jpeg', msk_type='png'
    ):

    X = np.array([
        image2mask(os.path.join(images_path, i), img_type) for i in instances])
    y = np.array([
        image2mask(os.path.join(masks_path, i), msk_type)for i in instances])
    y = y.reshape([*y.shape, 1])
    
    return X, y


def get_area(instance_path):
    return (gp.read_file(instance_path)['geometry'].area / 100).median()

    
def get_labels(distr):
    res = np.full(distr.shape, 3)
    res[distr < np.quantile(distr, 0.75)] = 2
    res[distr < np.quantile(distr, 0.5)] = 1
    res[distr < np.quantile(distr, 0.25)] = 0
    return res


def stratify(datasets_path, test_size):
    datasets = get_data_pathes(datasets_path)
    images_path, masks_path, instances_path = datasets[0]
    instances = list(os.walk(instances_path))[0][1]
    X, _ = get_data(images_path, masks_path, instances)
    areas = np.array([
        get_area(os.path.join(instances_path, i, i + '.geojson')) for i in instances])
    labels = get_labels(areas)

    sss = StratifiedShuffleSplit(
        n_splits=len(datasets), test_size=test_size, random_state=42)
    
    return sss.split(X, labels)


def build_stratified_generator(datasets_path, test_size=0.2, train=True):
    stratified_ix = stratify(datasets_path, test_size)
    datasets = get_data_pathes(datasets_path) 
    for i, (train_ix, test_ix) in enumerate(stratified_ix):
        images_path, masks_path, instances_path = datasets[i]
        instances = list(os.walk(instances_path))[0][1]
        X, y = get_data(images_path, masks_path, instances)
        if train:
            yield X[train_ix], y[train_ix]
        else:
            yield X[test_ix], y[test_ix]