import argparse
import os
import random
import re

import geopandas as gp
import numpy as np
import pandas as pd
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(
        description='Script for splitting data into train/test dataframes.')

    parser.add_argument('--datasets_path', '-dp', required=True, help='Path to the directory with all datasets')
    parser.add_argument('--markup_path', '-mp', required=True, help='Path to the original markup')
    parser.add_argument('--save_path', '-sp', required=True, help='Path to the save path')
    parser.add_argument('--folds', '-f', default=4, type=int, help='Number of folds')
    return parser.parse_args()


def get_image_info(instance):
    name_parts = re.split(r'[_.]', instance)
    return '_'.join(name_parts[:2]), '_'.join(name_parts[-3:-1])


def add_record(data_info, dataset_folder, name, position):
    return data_info.append(
        pd.DataFrame({
            'dataset_folder': dataset_folder,
            'name': name,
            'position': position
        }, index=[0]),
        sort=True, ignore_index=True
    )


def get_height_bounds(geometry):
    return geometry.total_bounds[1], geometry.total_bounds[3]


def update_overall_sizes(overall_sizes, test, train, val, deleted):
    overall_sizes["test"] += test
    overall_sizes["train"] += train
    overall_sizes["val"] += val
    overall_sizes["deleted"] += deleted
    return overall_sizes


def geo_split(datasets_path, markup_path, save_path, mask_type="png",
              test_height_threshold=0.3, val_height_bottom_threshold=0.3, val_height_top_threshold=0.4, fold=''):
    datasets = list(os.walk(datasets_path))[0][1]
    geojson_markup = gp.read_file(markup_path)

    minY, maxY = get_height_bounds(geojson_markup)

    height = maxY - minY

    cols = ['dataset_folder', 'name', 'position']
    train_df = pd.DataFrame(columns=cols)
    val_df = pd.DataFrame(columns=cols)
    test_df = pd.DataFrame(columns=cols)

    overall_sizes = {"test": 0, "train": 0, "val": 0, "deleted": 0}

    for dataset_dir in datasets:
        polys_path = os.path.join(datasets_path, dataset_dir, "geojson_polygons")
        print(dataset_dir)

        deleted = 0
        train = 0
        test = 0
        val = 0

        for poly_name in os.listdir(polys_path):
            instance_geojson_path = os.path.join(polys_path, poly_name)
            instance_geojson = gp.read_file(instance_geojson_path)

            if geojson_markup.crs != instance_geojson.crs:
                geojson_markup = geojson_markup.to_crs(instance_geojson.crs)
                minY, maxY = get_height_bounds(geojson_markup)
                height = maxY - minY

            instance_minY, instance_maxY = get_height_bounds(instance_geojson)

            name, position = get_image_info(poly_name)

            masks_path = os.path.join(datasets_path, dataset_dir, "masks")
            mask_path = os.path.join(masks_path, name + '_' + position + '.' + mask_type)
            mask = Image.open(mask_path)
            mask_array = np.array(mask)

            mask_pixels = np.count_nonzero(mask_array)
            center_pixels = np.count_nonzero(mask_array[10:-10, 10:-10])
            border_pixels = mask_pixels - center_pixels

            if mask_pixels > mask_array.size * 0.001 and center_pixels > border_pixels:
                if instance_maxY < minY + height * test_height_threshold:
                    test += 1
                    test_df = add_record(test_df, dataset_folder=name, name=name, position=position)
                elif instance_maxY < minY + height * val_height_top_threshold \
                        and instance_minY > minY + height * val_height_bottom_threshold:
                    val += 1
                    val_df = add_record(val_df, dataset_folder=name, name=name, position=position)
                else:
                    train += 1
                    train_df = add_record(train_df, dataset_folder=name, name=name, position=position)
            else:
                deleted += 1

        print("Train size", train, "Validation size", val, "Test size", test)
        print(f"{deleted} images were deleted")
        overall_sizes = update_overall_sizes(overall_sizes, test, train, val, deleted)

    print("Overall sizes", overall_sizes)

    train_df.to_csv(os.path.join(save_path, f'train{fold}.csv'), index=None, header=True)
    val_df.to_csv(os.path.join(save_path, f'val{fold}.csv'), index=None, header=True)
    test_df.to_csv(os.path.join(save_path, f'test{fold}.csv'), index=None, header=True)


def fold_split(datasets_path, markup_path, save_path, folds, test_height_threshold=0.3):
    for fold in range(folds):
        geo_split(datasets_path, markup_path, save_path,
                  val_height_top_threshold=1 - (1 - test_height_threshold) / folds * fold,
                  val_height_bottom_threshold=1 - (1 - test_height_threshold) / folds * (fold + 1),
                  test_height_threshold=test_height_threshold, fold=str(fold))


def autoencoder_split(datasets_path, markup_path, save_path,
                      test_height_threshold=0.3, val_height_threshold=0.4):
    datasets = list(os.walk(datasets_path))[0][1]
    geojson_markup = gp.read_file(markup_path)

    maxY = geojson_markup.total_bounds[3]
    minY = geojson_markup.total_bounds[1]

    height = maxY - minY

    cols = ['dataset_folder', 'name', 'position']
    train_df = pd.DataFrame(columns=cols)
    val_df = pd.DataFrame(columns=cols)
    test_df = pd.DataFrame(columns=cols)

    overall_sizes = {"test": 0, "train": 0, "val": 0, "deleted": 0}

    for dataset_dir in datasets:
        instances_path = os.path.join(datasets_path, dataset_dir, "geojson_polygons")
        print(dataset_dir)

        deleted = 0
        train = 0
        test = 0
        val = 0

        for instances_dir in os.listdir(instances_path):
            instance_geojson_path = os.path.join(instances_path, instances_dir)
            instance_geojson = gp.read_file(instance_geojson_path)

            if geojson_markup.crs != instance_geojson.crs:
                geojson_markup = geojson_markup.to_crs(instance_geojson.crs)

                maxY = geojson_markup.total_bounds[3]
                minY = geojson_markup.total_bounds[1]
                height = maxY - minY

            instance_maxY = instance_geojson.total_bounds[3]

            instance = instances_dir.split('_')
            name = '_'.join(instance[:2])
            position = '_'.join(instance[-2:]).split('.')[0]

            if instance_maxY < minY + height * test_height_threshold:
                test += 1
                test_df = add_record(test_df, dataset_folder=name, name=name, position=position)
            elif instance_maxY < minY + height * val_height_threshold:
                val += 1
                val_df = add_record(val_df, dataset_folder=name, name=name, position=position)
            else:
                train += 1
                train_df = add_record(train_df, dataset_folder=name, name=name, position=position)

        print("Train size", train)
        print("Validation size", val)
        print("Test size", test)
        print(f"{deleted} images was deleted")
        overall_sizes["test"] += test
        overall_sizes["train"] += train
        overall_sizes["val"] += val
        overall_sizes["deleted"] += deleted

    print("Overall sizes", overall_sizes)

    train_df.to_csv(os.path.join(save_path, 'train.csv'), index=None, header=True)
    val_df.to_csv(os.path.join(save_path, 'val.csv'), index=None, header=True)
    test_df.to_csv(os.path.join(save_path, 'test.csv'), index=None, header=True)


def train_val_split(datasets_path, save_path, train_val_ratio=0.3):
    random.seed(42)
    datasets = list(os.walk(datasets_path))[0][1]

    cols = ['dataset_folder', 'name', 'position']
    train_df = pd.DataFrame(columns=cols)
    val_df = pd.DataFrame(columns=cols)

    for dataset_dir in datasets:
        polys_path = os.path.join(datasets_path, dataset_dir, "geojson_polygons")
        print(dataset_dir)
        for poly_name in os.listdir(polys_path):
            name, position = get_image_info(poly_name)
            if random.random() <= train_val_ratio:
                val_df = add_record(val_df, dataset_folder=name, name=name, position=position)
            else:
                train_df = add_record(train_df, dataset_folder=name, name=name, position=position)
    train_df.to_csv(os.path.join(save_path, 'field_train.csv'), index=None, header=True)
    val_df.to_csv(os.path.join(save_path, 'field_val.csv'), index=None, header=True)


if __name__ == "__main__":
    args = parse_args()
    geo_split(args.datasets_path, args.markup_path, args.save_path)
