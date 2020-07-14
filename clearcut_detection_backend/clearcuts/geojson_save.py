import datetime
from datetime import date

import geopandas as gp
import numpy as np
from django.contrib.gis.geos import GEOSGeometry

from .models import Clearcut, Zone


def convert_geodataframe_to_geospolygons(dataframe):
    geometries = []

    for data in dataframe.to_dict("records"):
        geometry_str = data.pop('geometry')
        try:
            geometry = GEOSGeometry(str(geometry_str), srid=4326)
        except (TypeError, ValueError) as exc:
            print(exc)
            continue
        geometries.append(geometry)
    return geometries


def save_clearcut(poly, avg_area, detection_date, forest, not_forest, cloud, area_in_meters, zone=None, create_new_zone=False, area_threshold=0.2):
    if area_in_meters > avg_area * area_threshold and poly.geom_type == 'Polygon':
        if create_new_zone:
            zone = Zone()
            zone.save()
        clearcut = Clearcut(
            image_date_previous=detection_date[0],
            image_date_current=detection_date[1], 
            mpoly=poly, area=area_in_meters, zone=zone, 
            forest=forest,
            not_forest=not_forest,
            clouds=cloud,
            centroid=poly.centroid
        )
        clearcut.save()


def save(tile, poly_path, forest, not_forest, cloud, init_db=False):
    predicted_clearcuts = gp.read_file(poly_path)
    area_geodataframe = predicted_clearcuts['geometry'].area
    predicted_clearcuts = predicted_clearcuts.buffer(0).to_crs({'init': 'epsg:4326'})
    geodataframe = gp.GeoDataFrame(geometry=predicted_clearcuts)
    geospolygons = convert_geodataframe_to_geospolygons(geodataframe)
    avg_area = np.mean(area_geodataframe)

    '''
    if init_db:
        detection_date = [tile.first().tile_date, tile.last().tile_date]
    else:
        detection_date = [date.today(), date.today()]
    '''
    detection_date = [tile.first().tile_date, tile.last().tile_date]

    if Clearcut.objects.all().count() == 0:
        for idx, geopoly in enumerate(geospolygons):
            save_clearcut(geopoly, avg_area, detection_date,
                          forest, not_forest, cloud,
                          area_geodataframe[idx], create_new_zone=True)
    else:
        for idx, geopoly in enumerate(geospolygons):
            intersecting_polys = Clearcut.objects.filter(centroid__intersects=geopoly)

            if intersecting_polys.count() > 0:
                polys = [poly for poly in intersecting_polys]
                areas = [poly.mpoly.intersection(geopoly).area for poly in polys]
                max_intersection_area = np.argmax(areas)
                save_clearcut(geopoly, avg_area, detection_date,
                              forest, not_forest, cloud,
                              area_geodataframe[idx], zone=polys[max_intersection_area].zone)
            else:
                save_clearcut(geopoly, avg_area, detection_date,
                              forest, not_forest, cloud,
                              area_geodataframe[idx], create_new_zone=True)
