import argparse


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
        default='../output',
        help='Path to directory where data will be stored')
    parser.add_argument(
        '--width', '-w',  dest='width', default=224,
        type=int, help='Width of a piece')
    parser.add_argument(
        '--height', '-hgt', dest='height', default=224,
        type=int, help='Height of a piece')

    return parser.parse_args()


args = parse_args()
