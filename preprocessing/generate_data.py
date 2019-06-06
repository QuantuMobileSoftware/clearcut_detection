import os
import imageio
import argparse
import numpy as np
import pandas as pd
import geopandas as gp

from sklearn.model_selection import StratifiedShuffleSplit


def parse_args():
    parser = argparse.ArgumentParser(
        description='Script for creating tables that \
            contain info about all data pathes and names.'
    )
    parser.add_argument(
        '--data_path', '-dp', dest='data_path',
        default='../data', help='Path to the data'
    )
    parser.add_argument(
        '--save_path', '-sp', dest='save_path',
        default='../data',
        help='Path to directory where data will be stored'
    )
    parser.add_argument(
        '--images_folder', '-imf', dest='images_folder',
        default='images',
        help='Name of folder where images are storing'
    )
    parser.add_argument(
        '--masks_folder', '-mf', dest='masks_folder',
        default='masks',
        help='Name of folder where masks are storing'
    )
    parser.add_argument(
        '--instances_folder', '-inf', dest='instances_folder',
        default='instance_masks',
        help='Name of folder where instances are storing'
    )
    parser.add_argument(
        '--image_type', '-imt', dest='image_type',
        default='tiff',
        help='Type of image file'
    )
    parser.add_argument(
        '--mask_type', '-mt', dest='mask_type',
        default='png',
        help='Type of mask file'
    )
    parser.add_argument(
        '--instance_type', '-int', dest='instance_type',
        default='geojson',
        help='Type of instance file'
    )
    parser.add_argument(
        '--channels', '-ch', dest='channels',
        default=['rgb'],
        help='Channel list', type=list
    )

    return parser.parse_args()


args = parse_args()


def get_data_info(data_path=args.data_path):  
    
    _, _, insatnces_path = get_data_pathes(data_path)
    instances = get_folders(insatnces_path)
    
    cols = ['date', 'name', 'ix', 'iy']
    data_info = pd.DataFrame(columns=cols)
    for instance in instances:
        name_parts = split_fullname(instance)
        data_info = data_info.append(
            pd.DataFrame({
                'date': name_parts[0],
                'name': name_parts[1],
                'ix': name_parts[3],
                'iy': name_parts[4]
            }, index=[0]),
            sort=True, ignore_index=True
        )
        
    return data_info


def get_data_pathes(
    data_path=args.data_path, images_folder=args.images_folder,
    masks_folder=args.masks_folder, instances_folder=args.instances_folder
):
    
    dataset = get_folders(data_path)[0]
    
    images_path = os.path.join(data_path, dataset, images_folder)
    masks_path = os.path.join(data_path, dataset, masks_folder)
    insatnces_path = os.path.join(data_path, dataset, instances_folder)
    
    return images_path, masks_path, insatnces_path
    
    
def get_folders(path):
    return list(os.walk(path))[0][1]


def split_fullname(fullname):
    return fullname.split('_')


def get_fullname(*name_parts):
    return '_'.join(tuple(map(str, name_parts)))


def get_filepath(*path_parts, file_type):
    return '{}.{}'.format(join_pathes(*path_parts), file_type)


def join_pathes(*pathes):
    return os.path.join(*pathes)


def stratify(
    data_info, data_path=args.data_path, 
    test_size=0.2, random_state=42,
    channel=args.channels[0], instance_type=args.instance_type,
    instances_folder=args.instances_folder
):
    
    X, _ = get_data(data_info)
    areas = []
    for _, row in data_info.iterrows():
        instance_name = get_fullname(row['date'], row['name'], channel, row['ix'], row['iy'])
        instance_path = get_filepath(
            data_path,
            get_fullname(row['date'], row['name'], channel),
            instances_folder,
            instance_name,
            instance_name,
            file_type=instance_type
        )
        areas.append(get_area(instance_path))
                     
    labels = get_labels(np.array(areas))

    sss = StratifiedShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state
    )

    return sss.split(X, labels)


def get_data(
    data_info, channel=args.channels[0], data_path=args.data_path,
    image_folder=args.images_folder, mask_folder=args.masks_folder,
    image_type=args.image_type, mask_type=args.mask_type
):
    
    x = []
    y = []
    for _, row in data_info.iterrows():
        dataset = get_fullname(row['date'], row['name'], channel)
        filename = get_fullname(row['date'], row['name'], channel, row['ix'], row['iy'])
        
        image_path = get_filepath(
            data_path,
            dataset,
            image_folder,
            filename,
            file_type=image_type
        )
        mask_path = get_filepath(
            data_path,
            dataset,
            mask_folder,
            filename,
            file_type=mask_type
        )
        
        x.append(read_tensor(image_path))
        y.append(read_tensor(mask_path))
        
    x = np.array(x)
    y = np.array(y)
    y = y.reshape([*y.shape, 1])

    return x, y


def read_tensor(filepath):
    return imageio.imread(filepath)


def get_area(instance_path):
    return (gp.read_file(instance_path)['geometry'].area / 100).median()


def get_labels(distr):
    res = np.full(distr.shape, 3)
    res[distr < np.quantile(distr, 0.75)] = 2
    res[distr < np.quantile(distr, 0.5)] = 1
    res[distr < np.quantile(distr, 0.25)] = 0
    return res


def stratified_split(
    data_info, data_path=args.data_path,
    test_size=0.2, random_state=42,
    channel=args.channels[0], instance_type=args.instance_type,
    instances_folder=args.instances_folder
):
    
    stratified_indexes = stratify(
        data_info, data_path,
        test_size, random_state,
        channel, instance_type,
        instances_folder
    )
    
    for train_ix, test_ix in stratified_indexes:
        train_df = data_info.iloc[train_ix]
        test_df = data_info.iloc[test_ix]
    
    return train_df, test_df


if __name__ == '__main__':
    data_info = get_data_info()
    train_df, test_df = stratified_split(data_info)
    
    train_df.to_csv(
        get_filepath(args.save_path, 'train_df', file_type='csv'),
        index=False
    )
    test_df.to_csv(
        get_filepath(args.save_path, 'test_df', file_type='csv'),
        index=False
    )
