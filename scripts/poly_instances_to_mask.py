from rasterio import features
import os
import rasterio as rs
import geopandas as gp
import argparse
from tqdm import tqdm
import pandas as pd
from shapely.geometry import MultiPolygon
from itertools import combinations
from geopandas import GeoSeries
import imageio


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
        '--original_image', '-oi', dest='original_image',
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

    original_image = rs.open(original_image_path)
    geojson_markup = gp.read_file(markup_path)
    geojson_markup = geojson_markup.to_crs(original_image.crs)

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

        adjacency_list = compose_adjacency_list(intersection["geometry"])
        components = get_components(intersection["geometry"], adjacency_list)

        multi_polys = []
        for component in components:
            multi_polys.append(MultiPolygon(poly for poly in component))

        if len(multi_polys) > 0:
            gs = GeoSeries(multi_polys)
            gs.crs = original_image.crs
            piece_geojson_name = "{0}.geojson".format(filename)
            gs.to_file("{0}/{1}/{2}".format(save_path, filename, piece_geojson_name), driver='GeoJSON')

        original_transform = original_image.transform
        for idx, component in enumerate(components):
            mask = features.rasterize(
                shapes=component,
                out_shape=(width, height),
                transform=[original_transform[0], original_transform[1], x,
                           original_transform[3], original_transform[4], y])

            imageio.imwrite("{0}/{1}/{2}.png".format(save_path, filename, idx), mask)


def compose_adjacency_list(polys):
    length = len(polys)
    adjacency_list = [set() for x in range(0, length)]
    area_threshold = 20
    for idx_tuple in combinations(range(len(polys)), 2):
        poly1 = polys.iloc[idx_tuple[0]]
        poly2 = polys.iloc[idx_tuple[1]]
        if poly1.intersects(poly2):
            if poly1.buffer(1).intersection(poly2).area > area_threshold:
                adjacency_list[idx_tuple[0]].add(idx_tuple[1])
                adjacency_list[idx_tuple[1]].add(idx_tuple[0])
    return adjacency_list


def bfs(graph, start, visited):
    saved = visited.copy()
    queue = [start]
    while queue:
        vertex = queue.pop(0)
        if vertex not in visited:
            visited.add(vertex)
            queue.extend(graph[vertex] - visited)

    return visited.difference(saved)


def get_components(polys, adjacency_list):
    visited = set()
    graph_components = []
    for i in range(len(polys)):
        dif = bfs(adjacency_list, i, visited)
        if dif:
            graph_components.append(polys[list(dif)])
    return graph_components


if __name__ == "__main__":
    args = parse_args()
    markup_to_separate_polygons(args.geojson_pieces,
                                args.geojson_markup,
                                args.save_path,
                                args.pieces_info_path,
                                args.original_image)
