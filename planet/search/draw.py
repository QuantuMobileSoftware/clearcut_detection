from collections import defaultdict

import cv2
import numpy as np
import pyproj
from shapely.geometry import Polygon, MultiPolygon
from skimage.transform import ProjectiveTransform

from search.helper import pprint


def _to_from_utm(coords, zone, inv=False):
    coordinates = [coords.tolist()]
    proj = pyproj.Proj(proj="utm", zone=zone, ellps="WGS84", datum="WGS84")
    new_coordinates = [[proj(*point, inverse=inv) for point in ring] for ring in coordinates]

    return np.array(new_coordinates[0])


def _get_vertices(coordinates):
    minx = min(coordinates, key=lambda coordinate: coordinate[0] + coordinate[1])
    miny = min(coordinates, key=lambda coordinate: coordinate[0] - coordinate[1])
    maxx = max(coordinates, key=lambda coordinate: coordinate[0] + coordinate[1])
    maxy = max(coordinates, key=lambda coordinate: coordinate[0] - coordinate[1])

    return np.array([minx, miny, maxx, maxy, minx])


def _find_contours(image):
    # color filter, (0, 0, 0) - full black, start from (1, 1, 1)

    img = image.copy()
    hsv_min = np.array((1, 1, 1), np.uint8)
    hsv_max = np.array((255, 255, 255), np.uint8)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # tile pixels must be white, background - black
    img = cv2.inRange(hsv, hsv_min, hsv_max)
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # отображаем контуры поверх изображения
    # cv2.drawContours(image, contours, -1, (255, 0, 0), 3, cv2.LINE_AA, hierarchy, 1)
    # cv2.imwrite('img.png', image)

    return contours, hierarchy


def _contours_to_polygons(contours, hierarchy, min_area=10.):
    # Convert a mask ndarray (binarized image) to Multipolygons

    if not contours:
        raise RuntimeError("Not contours found")
    # now messy stuff to associate parent and child contours
    cnt_children = defaultdict(list)
    child_contours = set()

    assert hierarchy.shape[0] == 1

    for idx, (_, _, _, parent_idx) in enumerate(hierarchy[0]):
        if parent_idx != -1:
            child_contours.add(idx)
            cnt_children[parent_idx].append(contours[idx])
    # create actual polygons filtering by area (removes artifacts)
    all_polygons = []
    for idx, cnt in enumerate(contours):
        if idx not in child_contours and cv2.contourArea(cnt) >= min_area:
            assert cnt.shape[1] == 1
            poly = Polygon(shell=cnt[:, 0, :],
                           holes=[c[:, 0, :] for c in cnt_children.get(idx, [])
                                  if cv2.contourArea(c) >= min_area])
            all_polygons.append(poly)
    all_polygons = MultiPolygon(all_polygons)

    return all_polygons


def _simplify_contours(contours, hierarchy):

    polygons = _contours_to_polygons(contours, hierarchy)
    polygons = polygons.simplify(3).buffer(0)

    if isinstance(polygons, MultiPolygon):
        coordinates = [point for polygon in polygons
                       for point in polygon.exterior.coords]
    else:
        coordinates = [point for point in polygons.exterior.coords]

    return coordinates


def _find_transform_matrix(utm_coordinates, coordinates):
    coordinates = np.array([coordinates[1],
                            coordinates[0],
                            coordinates[3],
                            coordinates[2],
                            coordinates[1]
                            ])

    matrix = ProjectiveTransform()
    is_success = matrix.estimate(utm_coordinates, coordinates)
    if not is_success:
        raise ArithmeticError("Cannot find transform matrix for given inputs")
    return matrix.params


def _utm_to_pixels(utm_coordinates, matrix, width, thickness):
    # add 1.0 as third coordinate (x, y, 1)
    utm_coordinates = np.insert(utm_coordinates, 2, 1, axis=1)
    # apply transform matrix
    pixels = np.dot(matrix, utm_coordinates.T)
    # to homogeneous pixel coordinates
    pixels /= pixels[2]
    pixels = np.array([[x, y] for [x, y] in pixels[:2].T], dtype=np.int32)
    # set max visual bounds to pixels coordinates
    pixels[pixels < 0] = thickness
    pixels[pixels > width] = width - thickness

    return pixels


def _draw(coordinates, img_path, input_bbox, zone, thickness=3):
    image = cv2.imread(img_path)
    contours, hierarchy = _find_contours(image)

    simplified_coordinates = _simplify_contours(contours, hierarchy)
    pixel_coordinates = _get_vertices(simplified_coordinates)

    vertices = _get_vertices(coordinates)

    utm_coordinates = _to_from_utm(vertices, zone)
    input_bbox_utm = _to_from_utm(input_bbox, zone)

    matrix = _find_transform_matrix(utm_coordinates, pixel_coordinates)

    width = image.shape[0]
    pixel_polygon = _utm_to_pixels(utm_coordinates, matrix, width, thickness)
    pixel_bbox = _utm_to_pixels(input_bbox_utm, matrix, width, thickness)

    image = cv2.polylines(image.copy(), [pixel_polygon,
                                         pixel_bbox], True, (0, 255, 255), thickness)

    cv2.imwrite(img_path, image)


def draw_aoi(search_geometry, best_items, verbose):
    for item in best_items:
        coordinates = item["coordinates"]
        img_path = item["thumbnail_path"]
        input_bbox = np.asarray(search_geometry['coordinates'][0])
        item_id = item["id"]
        zone = int(item_id.split("_")[0][:-5])
        pprint(f"Drawing {item_id}...", verbose)
        try:
            _draw(coordinates, img_path, input_bbox, zone)
        except Exception as ex:
            pprint(f"Cannot draw AOI on {item['id']}, error: {str(ex)}", verbose)
