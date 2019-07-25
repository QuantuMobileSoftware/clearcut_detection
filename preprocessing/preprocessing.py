import os
import sys
import argparse
import rasterio
import numpy as np

from tqdm import tqdm
from image_division import divide_into_pieces
from binary_mask_converter import poly2mask, split_mask
from poly_instances_to_mask import markup_to_separate_polygons

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pytorch.utils import get_folders


def scale_img(img_file, min_value=0, max_value=255, output_type='Byte'):
    with rasterio.open(img_file) as src:
        img = src.read(1)
        img = np.nan_to_num(img)
        min_ = img.min()
        max_ = img.mean() + 2 * img.std()

        os.system(
            f'gdal_translate -ot {output_type} \
            -scale {min_} {max_} {min_value} {max_value} \
            {img_file} {os.path.splitext(img_file)[0]}_scaled.tif'
        )


def merge(save_path, *images):
    os.system(f'gdal_merge.py -separate -o {save_path} {" ".join(images)}')


def merge_bands(tiff_filepath, save_path, channels):
    for file in os.listdir(tiff_filepath):
        if file.endswith('.tif'):
            tiff_file = file
            break

    image_name = '_'.join(tiff_file.split('_')[:2])
    image_path = os.path.join(save_path, f'{image_name}.tif') 
    file_list = []

    for i, channel in enumerate(channels):
        img = os.path.join(tiff_filepath, '_'.join([image_name, channel]))
        file_list.append(f'{img}_scaled.tif')
        with rasterio.open(f'{img}.tif') as src:
            scale_img(f'{img}.tif')
            src.close()

    merge(image_path, *file_list)

    for file in file_list:
        os.remove(file)

    return image_path


def preprocess(
    tiff_path, save_path, width, height,
    polys_path, channels, type_filter,
    pxl_size_threshold
):
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
        print("Save directory created.")

    for tiff_name in get_folders(tiff_path):
        tiff_filepath = os.path.join(tiff_path, tiff_name)
        tiff_file = merge_bands(tiff_filepath, save_path, channels)

        data_path = os.path.join(save_path, tiff_file[:-4].split('/')[-1])
        divide_into_pieces(tiff_file, data_path, width, height)

        pieces_path = os.path.join(data_path, 'masks')
        pieces_info = os.path.join(data_path, 'image_pieces.csv')
        mask_path = poly2mask(polys_path, tiff_file, data_path, type_filter)
        split_mask(mask_path, pieces_path, pieces_info)

        geojson_polygons = os.path.join(data_path, "geojson_polygons")
        instance_masks_path = os.path.join(data_path, "instance_masks")
        markup_to_separate_polygons(
            poly_pieces_path=geojson_polygons, markup_path=polys_path,
            save_path=instance_masks_path, pieces_info_path=pieces_info,
            original_image_path=tiff_file,
            image_pieces_path=os.path.join(data_path, 'images'),
            mask_pieces_path=pieces_path,
            pxl_size_threshold=pxl_size_threshold
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description='Script for creating binary mask from geojson.'
    )
    parser.add_argument(
        '--polys_path', '-pp', dest='polys_path',
        required=True, help='Path to the polygons'
    )
    parser.add_argument(
        '--tiff_path', '-tp', dest='tiff_path',
        required=True, help='Path to directory with source tiff folders'
    )
    parser.add_argument(
        '--save_path', '-sp', dest='save_path',
        default='../data/input',
        help='Path to directory where data will be stored'
    )
    parser.add_argument(
        '--width', '-w',  dest='width', default=224,
        type=int, help='Width of a piece'
    )
    parser.add_argument(
        '--height', '-hgt', dest='height', default=224,
        type=int, help='Height of a piece'
    )
    parser.add_argument(
        '--channels', '-ch', dest='channels',
        default=['rgb', 'ndvi', 'b8'],
        nargs='+', help='Channels list'
    )
    parser.add_argument(
        '--type_filter', '-tf', dest='type_filter',
        help='Type of clearcut: "open" or "closed")'
    )
    parser.add_argument(
        '--pxl_size_threshold', '-mp', dest='pxl_size_threshold',
        default=20, help='Minimum pixel size of mask area'
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    preprocess(
        args.tiff_path, args.save_path,
        args.width, args.height,
        args.polys_path, args.channels,
        args.type_filter, args.pxl_size_threshold
    )
