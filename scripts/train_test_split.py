import argparse
import os
import pandas as pd
import geopandas as gp


def parse_args():
    parser = argparse.ArgumentParser(
        description='Script for creating binary mask from geojson.')
    parser.add_argument(
        '--datasets_path', '-dp', dest='datasets_path',
        default='../../test_data/preprocessed_2016/',
        help='Path to the directory with all datasets')
    parser.add_argument(
        '--markup_path', '-mp', dest='markup_path',
        default='../../test_data/geojson/clearcuts_backup_2016-01-03.geojson',
        help='Path to the original markup')
    parser.add_argument(
        '--save_path', '-sp', dest='save_path',
        default='.',
        help='Path to the save path')
    return parser.parse_args()


def split(datasets_path, markup_path, save_path):
    datasets = list(os.walk(datasets_path))[0][1]
    geojson_markup = gp.read_file(markup_path).to_crs({'init': 'epsg:32637'})

    maxY = geojson_markup.total_bounds[3]
    minY = geojson_markup.total_bounds[1]
    height = maxY - minY

    train_ratio = 0.7

    train_list = {"dataset_folder": [], "image_name": []}
    test_list = {"dataset_folder": [], "image_name": []}

    for dataset_dir in datasets:
        instances_path = os.path.join(datasets_path, dataset_dir, "instance_masks")
        print(dataset_dir)
        train = 0
        test = 0
        for instances_dir in os.listdir(instances_path):
            instance_geojson_path = os.path.join(instances_path, instances_dir, instances_dir + ".geojson")
            instance_geojson = gp.read_file(instance_geojson_path)

            instance_minY = instance_geojson.total_bounds[1]

            if instance_minY > minY + height * (1 - train_ratio):
                train += 1
                train_list["dataset_folder"].append(dataset_dir)
                train_list["image_name"].append(instances_dir)
            else:
                test += 1
                test_list["dataset_folder"].append(dataset_dir)
                test_list["image_name"].append(instances_dir)

        print("Train size", train)
        print("Test size", test)

    train_df = pd.DataFrame(train_list, columns=["dataset_folder", "image_name"])
    test_df = pd.DataFrame(test_list, columns=["dataset_folder", "image_name"])

    train_df.to_csv(os.path.join(save_path, 'train.csv'), index=None, header=True)
    test_df.to_csv(os.path.join(save_path, 'test.csv'), index=None, header=True)


if __name__ == "__main__":
    args = parse_args()
    split(args.datasets_path, args.markup_path, args.save_path)
