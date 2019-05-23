import os
import re
import imageio
import argparse
import numpy as np
import pandas as pd
import geopandas as gp
from sklearn.model_selection import StratifiedShuffleSplit


def get_data_pathes(
    datasets_path, images_path_name='images',
    masks_path_name='masks', instances_path_name='instance_masks'
    ):

    datasets = list(os.walk(datasets_path))[0][1]
    print(datasets)
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
    img_type='tiff', msk_type='png'
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


def stratify(datasets_path, test_size=0.2, random_state=42):
    images_path, masks_path, instances_path = get_data_pathes(datasets_path)[0]
    instances = list(os.walk(instances_path))[0][1]

    X, _ = get_data(images_path, masks_path, instances)
    areas = np.array([
        get_area(os.path.join(instances_path, i, i + '.geojson')) for i in instances])
    labels = get_labels(areas)

    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state)
    
    return sss.split(X, labels)


def get_data_info(dataset_path):
    cols = [
        'name', 'channel', 'position',
        'image_path', 'mask_path', 'instance_path',
        'image_type', 'mask_type'
    ]
    data_info = pd.DataFrame(columns=cols)
    dataset = get_data_pathes(dataset_path)
    for subset in dataset:
        images_path, masks_path, instances_path = subset
        instances = list(os.walk(instances_path))[0][1]
        image_type = list(os.walk(images_path))[0][2][0].split('.')[-1]
        mask_type = list(os.walk(masks_path))[0][2][0].split('.')[-1]

        for i, instance in enumerate(instances):
            instance = instance.split('_')
            name = '_'.join(instance[:2])
            channel = '_'.join(instance[2:-2])
            position = '_'.join(instance[-2:])

            data_info = data_info.append(
                pd.DataFrame({
                    'name': name,
                    'channel': channel,
                    'position': position,
                    'image_path': images_path,
                    'mask_path': masks_path,
                    'instance_path': instances_path,
                    'image_type': image_type,
                    'mask_type': mask_type
                }, index=[0]),
                sort=True, ignore_index=True)
    
    return data_info


def filter_by_channel(data_info, channel_name):
    return data_info[data_info['channel'] == channel_name] \
        .sort_values(by=['position']) \
        .reset_index(drop=True)


def build_batch_generator(
    data_info, indexes, batch_size=4,
    channels=['rgb', 'ndvi', 'ndvi_color', 'b2']
    ):
    
    if len(channels) == 0:
        raise Exception('You have to set at least 1 channel.')
        
    for i in indexes:
        images = []
        masks = []            
        for _ in range(batch_size):
            res_image = []
            for channel in channels:
                channel_data = filter_by_channel(data_info, channel)
                filename = '_'.join([
                    channel_data['name'].values[i],
                    channel,
                    channel_data['position'].values[i]])
                image_path = os.path.join(
                    channel_data['image_path'].values[i],
                    '{}.{}'.format(
                        filename,
                        channel_data['image_type'].values[i]))
                mask_path = os.path.join(
                    channel_data['mask_path'].values[i],
                    '{}.{}'.format(
                        filename,
                        channel_data['mask_type'].values[i]))
                
                image = imageio.imread(image_path)
                if image.ndim == 2:
                    image= image.reshape(*image.shape, 1)
                
                res_image.append(image)

            images.append(np.concatenate(res_image, axis=2))
            mask = imageio.imread(mask_path)
            mask = mask.reshape(*mask.shape, 1)
            masks.append(mask)
            
        masks = np.array(masks, np.float32)
        images = np.array(images, np.float32)

        yield images, masks



def parse_args():
    parser = argparse.ArgumentParser(
        description='Script for creating tables that \
            contain info about all data pathes and names.')
    parser.add_argument(
        '--data_path', '-dp', dest='data_path',
        required=True, help='Path to the data')
    parser.add_argument(
        '--save_path', '-sp', dest='save_path', 
        default='../data',
        help='Path to directory where data will be stored')
    parser.add_argument(
        '--test_size', '-ts',  dest='test_size', default=0.2,
        type=float, help='Percent of data that will be used for testing/validating')
    parser.add_argument(
        '--random_state', '-rs', dest='random_state', default=42,
        type=int, help='Random seed')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train_df, test_df = stratified_split(
        datasets_path=args.data_path,
        test_size=args.test_size,
        random_state=args.random_state)

    train_df.to_csv(os.path.join(args.save_path, 'train_df.csv'))
    test_df.to_csv(os.path.join(args.save_path, 'test_df.csv'))