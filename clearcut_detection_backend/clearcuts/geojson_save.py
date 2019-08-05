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

        if geometry.geom_type != 'Polygon':
            print('not polygon')
            continue
        else:
            geometries.append(geometry)
    return geometries


def save_clearcut(poly, avg_area, detection_date, area_in_meters, zone):
    if poly.area > avg_area:
        clearcut = Clearcut(
            image_date=detection_date, mpoly=poly, area=area_in_meters, zone=zone,
            centroid=poly.centroid
        )
        clearcut.save()


def save(poly_path, init_db=False):
    predicted_clearcuts = gp.read_file(poly_path)
    predicted_clearcuts = predicted_clearcuts.buffer(0).to_crs({'init': 'epsg:4326'})
    geodataframe = gp.GeoDataFrame(geometry=predicted_clearcuts)
    area_geodataframe = geodataframe.to_crs({'init': 'epsg:3827'})['geometry'].area
    geospolygons = convert_geodataframe_to_geospolygons(geodataframe)
    avg_area = np.mean([poly.area for poly in geospolygons])

    if init_db:
        date_part = poly_path.split('_')[4]
        detection_date = datetime.datetime.strptime(date_part[:8], '%Y%m%d')
        print(f'{detection_date} started')
    else:
        detection_date = date.today()

    if Clearcut.objects.all().count() == 0:
        first_zone = Zone()
        first_zone.save()
        for idx, geopoly in enumerate(geospolygons):
            save_clearcut(geopoly, avg_area, detection_date, area_geodataframe[idx], first_zone)
    else:
        for idx, geopoly in enumerate(geospolygons):
            intersecting_polys = Clearcut.objects.filter(centroid__intersects=geopoly)

            if intersecting_polys.count() > 0:
                polys = [poly for poly in intersecting_polys]
                areas = [poly.mpoly.intersection(geopoly).area for poly in polys]
                max_intersection_area = np.argmax(areas)
                save_clearcut(geopoly, avg_area, detection_date, area_geodataframe[idx],
                              polys[max_intersection_area].zone)
            else:
                new_zone = Zone()
                new_zone.save()
                save_clearcut(geopoly, avg_area, detection_date, area_geodataframe[idx], new_zone)

    print(f'{str(detection_date)} finished')
