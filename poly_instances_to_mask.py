import imageio
from rasterio import features
import os
import rasterio as rs
import geopandas as gp
import argparse
from tqdm import tqdm
import pandas as pd

WIDTH = 224
HEIGHT = 224
CRS = {'init': 'epsg:32637'}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Script for creating binary mask from geojson.')
    parser.add_argument(
        '--geojson_pieces', '-gp', dest='geojson_pieces',
        required=True, help='Path to the directory geojson polygons of image pieces')
    parser.add_argument(
        '--geojson_markup', '-gm', dest='geojson_markup',
        required=True, help='Path to the original geojson markup')
    parser.add_argument(
        '--save_path', '-sp', dest='save_path',
        required=True, help='Path to the directory where separate polygons will be stored')
    parser.add_argument(
        '--pieces_info_path', '-pi', dest='pieces_info_path',
        required=True, help='Path to the image pieces info')
    parser.add_argument(
        '--original_image', '-pi', dest='original_image',
        required=True, help='Path to the source tif image')
    return parser.parse_args()


def markup_to_separate_polygons(poly_pieces_path,
                                markup_path,
                                save_path,
                                pieces_info_path,
                                original_image_path):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
        print("Output directory created.")

    geojson_markup = gp.read_file(markup_path)
    geojson_markup = geojson_markup.to_crs(CRS)
    original_image = rs.open(original_image_path)

    pieces_info = pd.read_csv(pieces_info_path)

    for i in tqdm(range(len(pieces_info))):
        width = pieces_info["width"][i]
        height = pieces_info["height"][i]
        poly_piece_name = pieces_info['piece_geojson'][i]
        start_x = pieces_info["start_x"][i]
        start_y = pieces_info["start_y"][i]

        x, y = original_image.transform * (start_x + 1, start_y + 1)

        poly_piece = gp.read_file(os.path.join(poly_pieces_path, poly_piece_name))
        intersection = gp.overlay(geojson_markup, poly_piece, how='intersection')

        filename, file_extension = os.path.splitext(poly_piece_name)
        if not os.path.exists(os.path.join(save_path, filename)):
            os.mkdir(os.path.join(save_path, filename))

        original_transform = original_image.transform

        for i, poly in enumerate(intersection["geometry"]):
            mask = features.rasterize(
                shapes=[poly],
                out_shape=(width, height),
                transform=[original_transform[0], original_transform[1], x,
                           original_transform[3], original_transform[4], y])

            imageio.imwrite(f"{save_path}/{filename}/{i}.png", mask)


if __name__ == "__main__":
    args = parse_args()
    markup_to_separate_polygons(args.geojson_pieces,
                                args.geojson_markup,
                                args.save_path,
                                args.pieces_info_path,
                                args.original_image)
