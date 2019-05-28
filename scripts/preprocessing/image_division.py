import rasterio
from rasterio.windows import Window
from rasterio.plot import reshape_as_image
from PIL import Image
import csv
import os
from tqdm import tqdm
import argparse
from shapely.geometry import Polygon
from geopandas import GeoSeries


def parse_args():
    parser = argparse.ArgumentParser(description='Script for dividing images into smaller pieces.')
    parser.add_argument('--image_path', '-ip', dest='image_path', required=True,
                        help='Path to source image')
    parser.add_argument('--save_path', '-sp', dest='save_path', default='../../data',
                        help='Path to directory where pieces will be stored')
    parser.add_argument('--width', '-w', dest='width', default=224, type=int,
                        help='Width of a piece')
    parser.add_argument('--height', '-hgt', dest='height', default=224, type=int,
                        help='Height of a piece')

    return parser.parse_args()


def divide_into_pieces(image_path, save_path, width, height):
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
        print("Data directory created.")

    os.makedirs("{0}/images".format(save_path), exist_ok=True)
    os.makedirs("{0}/geojson_polygons".format(save_path), exist_ok=True)
    with rasterio.open(image_path) as src, open('{0}/image_pieces.csv'.format(save_path), 'w') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow([
            'original_image', 'piece_image', 'piece_geojson',
            'start_x', 'start_y', 'width', 'height'])

        for j in tqdm(range(0, src.height // height)):
            for i in range(0, src.width // width):
                raster_window = src.read(
                    window=Window(i * width, j * height, width, height))

                filename_w_ext = os.path.basename(image_path)
                filename, _ = os.path.splitext(filename_w_ext)
                image_format = "tiff"
                piece_name = "{}_{}_{}.{}".format(filename, j, i, image_format)

                poly = Polygon([
                    src.xy(j * height, i * width),
                    src.xy(j * height, (i + 1) * width),
                    src.xy((j + 1) * height, (i + 1) * width),

                    src.xy((j + 1) * height, i * width),
                    src.xy(j * height, i * width)
                ])
                gs = GeoSeries([poly])
                gs.crs = src.crs
                piece_geojson_name = "{}_{}_{}.geojson".format(filename, j, i)
                gs.to_file("{}/geojson_polygons/{}".format(
                    save_path, piece_geojson_name),
                    driver='GeoJSON')

                image_array = reshape_as_image(raster_window)

                channels = image_array.shape[2]

                if channels == 1:
                    if image_array.dtype == 'float32':
                        image = Image.fromarray(image_array.reshape((224, 224)), mode='F')
                    elif image_array.dtype == 'uint16':
                        image = Image.fromarray(image_array.reshape((224, 224)), mode='I;16')
                elif channels == 3:
                    image = Image.fromarray(image_array, mode='RGB')
                elif channels == 4:
                    image = Image.fromarray(image_array, mode='RGBA')
                else:
                    print("Wrong format")
                    continue

                image.save("{}/images/{}".format(save_path, piece_name))

                writer.writerow([
                    filename_w_ext, piece_name, piece_geojson_name,
                    i * width, j * height, width, height])

    csvFile.close()


if __name__ == "__main__":
    args = parse_args()
    divide_into_pieces(args.image_path, args.save_path, args.width, args.height)
