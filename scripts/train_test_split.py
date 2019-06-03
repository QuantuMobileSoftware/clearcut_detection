import argparse
import os
from os.path import basename, normpath

import geopandas as gp
import pandas as pd
import sklearn


def parse_args():
    parser = argparse.ArgumentParser(
        description='Script for splitting data into train/test dataframes.')

    parser.add_argument('--datasets_path', '-dp', required=True, help='Path to the directory with all datasets')
    parser.add_argument('--markup_path', '-mp', required=True, help='Path to the original markup')
    parser.add_argument('--save_path', '-sp', required=True, help='Path to the save path')
    parser.add_argument('--image_size', '-is', default=224, type=int, help='Image size')
    return parser.parse_args()


def season_split(datasets_path, markup_path, save_path, img_size=224, mask_type="png", img_type="tiff"):
    datasets = list(os.walk(datasets_path))[0][1]
    geojson_markup = gp.read_file(markup_path).to_crs({'init': 'epsg:32637'})

    maxY = geojson_markup.total_bounds[3]
    minY = geojson_markup.total_bounds[1]
    height = maxY - minY

    train_ratio = 0.7

    train_list = {"dataset_folder": [], "name": [], "channel": [], "image_size": [], "mask_type": [],
                  "image_type": [], "position": []}
    test_list = {"dataset_folder": [], "name": [], "channel": [], "image_size": [], "mask_type": [],
                 "image_type": [], "position": []}

    for dataset_dir in datasets:
        instances_path = os.path.join(datasets_path, dataset_dir, "instance_masks")
        print(dataset_dir)

        train = 0
        test = 0

        for instances_dir in os.listdir(instances_path):
            instance_geojson_path = os.path.join(instances_path, instances_dir, instances_dir + ".geojson")
            instance_geojson = gp.read_file(instance_geojson_path)

            instance_minY = instance_geojson.total_bounds[1]

            instance = instances_dir.split('_')
            name = '_'.join(instance[:2])
            channel = '_'.join(instance[2:-2])
            position = '_'.join(instance[-2:])

            if instance_minY > minY + height * (1 - train_ratio):
                train += 1
                train_list["dataset_folder"].append(dataset_dir)
                train_list["name"].append(name)
                train_list["channel"].append(channel)
                train_list["position"].append(position)
                train_list["image_size"].append(img_size)
                train_list["mask_type"].append(mask_type)
                train_list["image_type"].append(img_type)
            else:
                test += 1
                test_list["dataset_folder"].append(dataset_dir)
                test_list["name"].append(name)
                test_list["channel"].append(channel)
                test_list["position"].append(position)
                test_list["image_size"].append(img_size)
                test_list["mask_type"].append(mask_type)
                test_list["image_type"].append(img_type)

        print("Train size", train)
        print("Test size", test)

    train_df = pd.DataFrame(train_list,
                            columns=["dataset_folder", "name", "channel", "image_size", "mask_type", "image_type",
                                     "position"])
    test_df = pd.DataFrame(test_list,
                           columns=["dataset_folder", "name", "channel", "image_size", "mask_type", "image_type",
                                    "position"])

    train_df.to_csv(os.path.join(save_path, 'train.csv'), index=None, header=True)
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
