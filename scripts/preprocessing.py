import os
import argparse

from image_division import divide_into_pieces
from binary_mask_converter import poly2mask, split_mask
from poly_instances_to_mask import markup_to_separate_polygons


def parse_args():
    parser = argparse.ArgumentParser(
        description='Script for creating binary mask from geojson.')
    parser.add_argument(
        '--polys_path', '-pp', dest='polys_path',
        required=True, help='Path to the polygons')
    parser.add_argument(
        '--tiff_path', '-tp', dest='tiff_path',
        required=True, help='Path to directory with source tiff files')
    parser.add_argument(
        '--save_path', '-sp', dest='save_path', 
        default='../data',
        help='Path to directory where data will be stored')
    parser.add_argument(
        '--width', '-w',  dest='width', default=224,
        type=int, help='Width of a piece')
    parser.add_argument(
        '--height', '-hgt', dest='height', default=224,
        type=int, help='Height of a piece')

    return parser.parse_args()


def preprocess(tiff_path, save_path, width, height, polys_path):
    for root, _, files in os.walk(tiff_path):
        for file in files:
            if file[-4:] != '.tif':
                continue
            tiff = os.path.join(root, file)
            image_path = os.path.join(save_path, file[:-4])
            divide_into_pieces(tiff, image_path, width, height)

            pieces_path = os.path.join(image_path, "masks")
            pieces_info = os.path.join(image_path, "image_pieces.csv")
            mask_path = poly2mask(polys_path, tiff, image_path)
            split_mask(mask_path, pieces_path, pieces_info)

            geojson_polygons = os.path.join(image_path, "geojson_polygons")
            instance_masks_path = os.path.join(image_path, "instance_masks")
            markup_to_separate_polygons(
                poly_pieces_path=geojson_polygons, markup_path=polys_path,
                save_path=instance_masks_path, pieces_info_path=pieces_info,
                original_image_path=tiff,
                image_pieces_path=os.path.join(image_path, 'images'),
                mask_pieces_path=pieces_path)


def stratify():
    pass

    
if __name__ == '__main__':
    args = parse_args()
    preprocess(
        args.tiff_path, args.save_path,
        args.width, args.height, args.polys_path)
