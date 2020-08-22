import logging
import time
import geopandas as gp
import numpy as np
from django.contrib.gis.geos import GEOSGeometry
from django.contrib.gis.measure import D
from .models import Clearcut, Zone, RunUpdateTask, NotClearcut

from django.contrib.gis.db.models.functions import Distance

SEARCH_WINDOW = 50

logger = logging.getLogger('update')


def convert_geodataframe_to_geospolygons(dataframe):
    geometries = []
    for data in dataframe.to_dict("records"):
        geometry_str = data.pop('geometry')
        try:
            geometry = GEOSGeometry(str(geometry_str), srid=4326)
        except (TypeError, ValueError):
            logger.error(f'GEOSGeometry error with geometry_str: {str(geometry_str)}')
            continue
        geometries.append(geometry)
    return geometries


def save_not_clearcut(
        poly,
        avg_area,
        detection_date,
        forest,
        cloud,
        area_in_meters,
        tile=None,
        area_threshold=0.2,
):
    if area_in_meters > avg_area * area_threshold and poly.geom_type == 'Polygon':
        zone = Zone()
        zone.tile = tile
        zone.save()
        not_clearcut = NotClearcut(
            image_date_previous=detection_date[0],
            image_date_current=detection_date[1],
            mpoly=poly, area=area_in_meters, zone=zone,
            forest=forest,
            clouds=cloud,
            centroid=poly.centroid
        )
        not_clearcut.save()
        logger.info(f'not_clearcut saved with id={not_clearcut.id}')


def save_clearcut(
        poly,
        avg_area,
        detection_date,
        forest, cloud,
        area_in_meters,
        zone=None,
        create_new_zone=False,
        area_threshold=0.2,
        tile=None,
        ):
    if area_in_meters > avg_area * area_threshold and poly.geom_type == 'Polygon':
        if create_new_zone:
            zone = Zone()
            zone.tile = tile
            zone.save()
        clearcut = Clearcut(
            image_date_previous=detection_date[0],
            image_date_current=detection_date[1],
            mpoly=poly, area=area_in_meters, zone=zone,
            forest=forest,
            clouds=cloud,
            centroid=poly.centroid
        )
        clearcut.save()
        logger.info(f'clearcut saved with id={clearcut.id}')


def save_from_task(task_id):
    logger.info(f'task_id: {task_id}')
    start_time = time.time()
    task = RunUpdateTask.objects.get(id=task_id)
    detection_date = [task.image_date_0, task.image_date_1]
    logger.info(f'detection_date: {detection_date}')
    predicted_clearcuts = gp.read_file(task.result)
    logger.info(f'opened: {task.result}')
    area_geodataframe = predicted_clearcuts['geometry'].area
    predicted_polys = predicted_clearcuts['geometry'].buffer(0).to_crs({'init': 'epsg:4326'})

    geodataframe = gp.GeoDataFrame(geometry=predicted_polys)
    flags_forest = predicted_clearcuts['forest'].to_list()
    flags_clouds = predicted_clearcuts['clouds'].to_list()
    geospolygons = convert_geodataframe_to_geospolygons(geodataframe)

    avg_area = np.mean(area_geodataframe)

    for idx, geopoly in enumerate(geospolygons):
        forest = flags_forest[idx]
        cloud = flags_clouds[idx]

        if forest == 0 or cloud == 1:
            save_not_clearcut(
                geopoly,
                avg_area,
                detection_date,
                flags_forest[idx],
                flags_clouds[idx],
                area_geodataframe[idx],
                tile=task.tile,
            )
            continue

        intersecting_polys = list(Clearcut.objects.filter(
            zone__tile_id=task.tile_id,
            mpoly__dwithin=(geopoly, D(m=SEARCH_WINDOW), 'spheroid'),
            ))
        if len(intersecting_polys) == 0:
            save_clearcut(geopoly,
                          avg_area,
                          detection_date,
                          forest,
                          cloud,
                          area_geodataframe[idx],
                          create_new_zone=True,
                          tile=task.tile,
                          )
        else:
            areas = []
            dates = []
            for poly in intersecting_polys:
                areas.append(poly.mpoly.intersection(geopoly).area)
                dates.append(poly.image_date_previous)
                geopoly = geopoly.union(poly.mpoly)

            max_intersection_area = np.argmax(areas)
            detection_date_union = [task.image_date_0, min(dates)]

            save_clearcut(geopoly,
                          avg_area,
                          detection_date_union,
                          forest,
                          cloud,
                          area_geodataframe[idx],
                          zone=intersecting_polys[max_intersection_area].zone,
                          tile=task.tile,
                          )
    logger.info(f'---{time.time() - start_time} seconds --- for saving task_id: {task_id}')
