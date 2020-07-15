import datetime
from datetime import date

import geopandas as gp
import numpy as np
from django.contrib.gis.geos import GEOSGeometry

from .models import Clearcut, Zone

'''
def convert_geodataframe_to_geospolygons(dataframe):
    geometries = []
    flags_clouds = []
    flags_forest = []
    for data in dataframe.to_dict("records"):
        geometry_str = data.pop('geometry')
        flag_cloud = data.pop('clouds')
        flag_forest = data.pop('forest')
        geometry = GEOSGeometry('POINT EMPTY', srid=4326)
        try:
            geometry = GEOSGeometry(str(geometry_str), srid=4326)
        except (TypeError, ValueError) as exc:
            # print(exc)
            continue
        geometries.append(geometry)
        flags_clouds.append(flag_cloud)
        flags_forest.append(flag_forest)
    return geometries, flags_clouds, flags_forest
'''
def convert_geodataframe_to_geospolygons(dataframe):
    geometries = []
    for data in dataframe.to_dict("records"):
        geometry_str = data.pop('geometry')
        try:
            geometry = GEOSGeometry(str(geometry_str), srid=4326)
        except (TypeError, ValueError) as exc:
            # print(exc)
            continue
        geometries.append(geometry)
    return geometries


def save_clearcut(poly, avg_area, detection_date, forest, cloud, area_in_meters, zone=None, create_new_zone=False, area_threshold=0.2):
    if area_in_meters > avg_area * area_threshold and poly.geom_type == 'Polygon':
        if create_new_zone:
            zone = Zone()
            zone.save()
        clearcut = Clearcut(
            image_date_previous=detection_date[1],
            image_date_current=detection_date[0], 
            mpoly=poly, area=area_in_meters, zone=zone, 
            forest=forest,
            clouds=cloud,
            centroid=poly.centroid
        )
        clearcut.save()


def save(tile, poly_path, init_db=False):
    predicted_clearcuts = gp.read_file(poly_path)
    area_geodataframe = predicted_clearcuts['geometry'].area
    predicted_polys = predicted_clearcuts['geometry'].buffer(0).to_crs({'init': 'epsg:4326'})

    geodataframe = gp.GeoDataFrame(geometry=predicted_polys)
    flags_forest = predicted_clearcuts['forest'].to_list()
    flags_clouds = predicted_clearcuts['clouds'].to_list()
    geospolygons = convert_geodataframe_to_geospolygons(geodataframe)
    
    avg_area = np.mean(area_geodataframe)
    detection_date = [tile.first().tile_date, tile.last().tile_date]

    if Clearcut.objects.all().count() == 0:
            save_clearcut(geopoly, avg_area, detection_date,
                          flags_forest[idx], flags_clouds[idx],
                          area_geodataframe[idx], create_new_zone=True)
    else:
        for idx, geopoly in enumerate(geospolygons):
            intersecting_polys = Clearcut.objects.filter(centroid__intersects=geopoly)
            forest = flags_forest[idx]
            cloud = flags_clouds[idx]
            if intersecting_polys.count() > 0:
                polys = [poly for poly in intersecting_polys]
                areas = [poly.mpoly.intersection(geopoly).area for poly in polys]
                dates = [poly.image_date_previous for poly in polys]
                max_intersection_area = np.argmax(areas)
                # Union of intersecting forest polygons:
                detection_date_union = detection_date
                if flags_forest[idx] == 1 and flags_clouds[idx] == 0:
                    for poly in polys:
                        geopoly = geopoly.union(poly.mpoly)
                    detection_date_union = [tile.first().tile_date, min(dates)]
                save_clearcut(geopoly, avg_area, detection_date_union,
                              flags_forest[idx], flags_clouds[idx],
                              area_geodataframe[idx], zone=polys[max_intersection_area].zone)
            else:
                save_clearcut(geopoly, avg_area, detection_date,
                              flags_forest[idx], flags_clouds[idx],
                              area_geodataframe[idx], create_new_zone=True)
