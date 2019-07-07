import os
import argparse
import numpy as np
import rasterio

from tqdm import tqdm
from image_division import divide_into_pieces
from binary_mask_converter import poly2mask, split_mask
from poly_instances_to_mask import markup_to_separate_polygons

from pytorch.utils import get_folders


def scale(tensor, value=1):
    while tensor.ndim < 3:
        tensor = np.expand_dims(tensor, -1)

    for i in range(tensor.shape[2]):
        max_value = tensor[:, :, i].max()
        if max_value not in [0, value]:
            tensor[:, :, i] = tensor[:, :, i] / max_value * value

    return tensor


def merge_bands(tiff_filepath, save_path, channels):
    for root, _, files in os.walk(tiff_filepath):
        tiff_file = [file for file in files if file[-4:] == '.tif'][0]
        image_name = '_'.join(tiff_file.split('_')[:2])
        image_path = os.path.join(
            save_path,
            f'{image_name}.tif'
        )
        file_list = []
        for i, channel in enumerate(channels):
            file_list.append(os.path.join(root, f"{'_'.join([image_name, channel])}.tif"))
            with rasterio.open(file_list[i]) as src:
                print(channel, src.meta['count'])
                if i == 0:
                    meta = src.meta
                else:
                    meta['count'] += src.meta['count']
                src.close()

        meta['dtype'] = np.uint8
        with rasterio.open(image_path, 'w', **meta) as dst:
            i = 1
            for layer in tqdm(file_list):
                with rasterio.open(layer) as src:
                    for j in range(1, src.meta['count'] + 1):
                        tensor = scale(src.read(j), 255)[:, :, 0].astype(np.uint8)
                        dst.write_band(i, tensor)
                        i += 1
                    src.close()
            dst.close()

        return image_path


def preprocess(
    tiff_path, save_path, width, height,
    polys_path, channels, type_filter,
    pxl_size_threshold
):
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
        print("Save directory created.")

    tiff_names = get_folders(tiff_path)

    for tiff_name in tiff_names:
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
        '--channels', '-ch', dest='channels', default=[
            'rgb', 'ndvi', 'ndvi_color',
            'b2', 'b3', 'b4', 'b8'
        ], nargs='+', help='Channels list'
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
