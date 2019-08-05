import geopandas as gp

from datetime import date
from django.contrib.gis.geos import GEOSGeometry, GeometryCollection

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


def convert(poly_path):
    db_clearcuts = Clearcut.objects.values_list('mpoly', flat=True)
    predicted_clearcuts = gp.read_file(poly_path)
    predicted_clearcuts = predicted_clearcuts.to_crs({'init': 'epsg:4326'})
    result = convert_geodataframe_to_geospolygons(predicted_clearcuts)
    for poly in result:
        clearcut = Clearcut(
            forest_type='', forest_state='', detected_class='',
            image_date=date.today(), mpoly=poly
        )
        clearcut.save()
