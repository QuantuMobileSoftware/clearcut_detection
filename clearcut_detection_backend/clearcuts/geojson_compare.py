from datetime import date

import geopandas as gp
from django.contrib.gis.geos import GEOSGeometry, GeometryCollection

from .models import Clearcut


def geojson_compare(stored_geometry, predicted_geometry, area_threshold=0.2):
    converted_geometry = GeometryCollection(list(stored_geometry))

    detected_clearcuts = convert_geodataframe_to_geospolygons(predicted_geometry)
    detected_clearcuts_geometry = GeometryCollection(detected_clearcuts)

    candidates = detected_clearcuts_geometry.buffer(0).difference(converted_geometry.buffer(0))
    result = []
    avg_clearcut_area = detected_clearcuts_geometry.area / len(detected_clearcuts_geometry)
    for candidate in candidates:
        if candidate.area > area_threshold * avg_clearcut_area:
            result.append(candidate)
    return result


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


def compare(poly_path):
    db_clearcuts = Clearcut.objects.values_list('mpoly', flat=True)
    predicted_clearcuts = gp.read_file(poly_path)
    predicted_clearcuts = predicted_clearcuts.to_crs({'init': 'epsg:4326'})
    result = geojson_compare(db_clearcuts, predicted_clearcuts)
    for poly in result:
        clearcut = Clearcut(
            forest_type='', forest_state='', detected_class='',
            image_date=date.today(), mpoly=poly
        )
        clearcut.save()
