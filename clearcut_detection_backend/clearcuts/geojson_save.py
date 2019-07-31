import datetime

import geopandas as gp
from django.contrib.gis.geos import GEOSGeometry

from .models import Clearcut


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


def save(poly_path):
    predicted_clearcuts = gp.read_file(poly_path)
    predicted_clearcuts = predicted_clearcuts.buffer(0).to_crs({'init': 'epsg:4326'})
    geodataframe = gp.GeoDataFrame(geometry=predicted_clearcuts)
    result = convert_geodataframe_to_geospolygons(geodataframe)
    date_part = poly_path.split('_')[4]
    date_time_obj = datetime.datetime.strptime(date_part[:8], '%Y%m%d')
    print(date_part + ' started')
    overall_area = 0
    for poly in result:
        overall_area += poly.area
    avg_area = overall_area / len(result)

    for poly in result:
        if poly.area > avg_area:
            clearcut = Clearcut(
                forest_type='', forest_state='', detected_class='',
                image_date=date_time_obj, mpoly=poly
            )
            clearcut.save()
    print(date_part + ' finished')
