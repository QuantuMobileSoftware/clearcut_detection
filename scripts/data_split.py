import argparse
import os
from os.path import basename, normpath

import geopandas as gp
import numpy as np
import pandas as pd
import sklearn
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(
        description='Script for splitting data into train/test dataframes.')

    parser.add_argument('--datasets_path', '-dp', required=True, help='Path to the directory with all datasets')
    parser.add_argument('--markup_path', '-mp', required=True, help='Path to the original markup')
    parser.add_argument('--save_path', '-sp', required=True, help='Path to the save path')
    parser.add_argument('--image_size', '-is', default=224, type=int, help='Image size')
    return parser.parse_args()


def add_record(dataframe, dataset_dir, name, channel, position, img_size, mask_type, img_type):
    return dataframe.append(
        pd.DataFrame({
            'dataset_folder': dataset_dir,
            'name': name,
            'channel': channel,
            'position': position,
            'image_size': img_size,
            'mask_type': mask_type,
            'image_type': img_type
        }, index=[0]),
        sort=True, ignore_index=True)


def season_split(datasets_path, markup_path, save_path, img_size=224, mask_type="png", img_type="tiff",
                 test_height_threshold=0.3, val_height_threshold=0.4):
    datasets = list(os.walk(datasets_path))[0][1]
    geojson_markup = gp.read_file(markup_path)

    maxY = geojson_markup.total_bounds[3]
    minY = geojson_markup.total_bounds[1]

    height = maxY - minY

    cols = ["dataset_folder", "name", "channel", "image_size", "mask_type", "image_type", "position"]
    train_df = pd.DataFrame(columns=cols)
    val_df = pd.DataFrame(columns=cols)
    test_df = pd.DataFrame(columns=cols)

    overall_sizes = {"test": 0, "train": 0, "val": 0, "deleted": 0}

    for dataset_dir in datasets:
        instances_path = os.path.join(datasets_path, dataset_dir, "instance_masks")

        print(dataset_dir)

        deleted = 0
        train = 0
        test = 0
        val = 0

        for instances_dir in os.listdir(instances_path):
            instance_geojson_path = os.path.join(instances_path, instances_dir, instances_dir + ".geojson")
            instance_geojson = gp.read_file(instance_geojson_path)

            if geojson_markup.crs != instance_geojson.crs:
                geojson_markup = geojson_markup.to_crs(instance_geojson.crs)

                maxY = geojson_markup.total_bounds[3]
                minY = geojson_markup.total_bounds[1]
                height = maxY - minY

            instance_maxY = instance_geojson.total_bounds[3]

            instance = instances_dir.split('_')
            name = '_'.join(instance[:2])
            channel = '_'.join(instance[2:-2])
            position = '_'.join(instance[-2:])

            masks_path = os.path.join(datasets_path, dataset_dir, "masks")

            mask_path = os.path.join(masks_path, name + '_' + channel + '_' + position + '.' + mask_type)

            mask = Image.open(mask_path)

            mask_array = np.array(mask)

            mask_pixels = np.count_nonzero(mask_array)
            center_pixels = np.count_nonzero(mask_array[10:-10, 10:-10])
            border_pixels = mask_pixels - center_pixels

            if mask_pixels > mask_array.size * 0.001 and center_pixels > border_pixels:
                if instance_maxY < minY + height * test_height_threshold:
                    test += 1
                    test_df = add_record(test_df, dataset_dir, name, channel, position, img_size, mask_type, img_type)
                elif instance_maxY < minY + height * val_height_threshold:
                    val += 1
                    val_df = add_record(val_df, dataset_dir, name, channel, position, img_size, mask_type, img_type)
                else:
                    train += 1
                    train_df = add_record(train_df, dataset_dir, name, channel, position, img_size, mask_type, img_type)
            else:
                deleted += 1

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


def train_test_split(dataset_path, save_path, train_test_ratio=0.8):
    masks_path = os.path.join(dataset_path, "masks")
    image_names = []
    for file in os.listdir(masks_path):
        if ".png" in file and ".xml" not in file:
            filename, file_extension = os.path.splitext(file)
            image_names.append(filename)

    filenames = sklearn.utils.shuffle(image_names, random_state=42)

    dataset_size = len(filenames)

    train_size = int(dataset_size * train_test_ratio)
    test_size = dataset_size - train_size

    print("Dataset size: {0}".format(dataset_size))
    print("Train size: {0}".format(train_size))
    print("Test size: {0}".format(test_size))

    output_name = basename(normpath(dataset_path))

    folder_names_train = [output_name] * train_size
    folder_names_test = [output_name] * test_size

    train_df = pd.DataFrame(zip(folder_names_train, filenames[:train_size]), columns=["dataset_folder", "image_name"])
    test_df = pd.DataFrame(zip(folder_names_test, filenames[train_size:]), columns=["dataset_folder", "image_name"])

    train_df.to_csv(os.path.join(save_path, '{0}_train.csv'.format(output_name)), index=None, header=True)
    test_df.to_csv(os.path.join(save_path, '{0}_test.csv'.format(output_name)), index=None, header=True)


if __name__ == "__main__":
    args = parse_args()
    season_split(args.datasets_path, args.markup_path, args.save_path, args.image_size)
