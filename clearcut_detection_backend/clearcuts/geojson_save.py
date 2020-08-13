import logging
import time
import geopandas as gp
import numpy as np
from django.contrib.gis.geos import GEOSGeometry
from django.contrib.gis.measure import D
from .models import Clearcut, Zone, RunUpdateTask

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


def save_clearcut(poly,
                  avg_area,
                  detection_date,
                  forest, cloud,
                  area_in_meters,
                  zone=None,
                  create_new_zone=False,
                  area_threshold=0.2,
                  tile_index=None,
                  ):
    if area_in_meters > avg_area * area_threshold and poly.geom_type == 'Polygon':
        if create_new_zone:
            zone = Zone()
            zone.tile_index = tile_index
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

    if Clearcut.objects.all().count() == 0:
        for idx, geopoly in enumerate(geospolygons):
            save_clearcut(geopoly,
                          avg_area,
                          detection_date,
                          flags_forest[idx],
                          flags_clouds[idx],
                          area_geodataframe[idx],
                          create_new_zone=True,
                          tile_index=task.tile_index,
                          )
    else:
        for idx, geopoly in enumerate(geospolygons):

            intersecting_polys = Clearcut.objects.filter(zone__tile_index_id=task.tile_index_id,
                                                         mpoly__distance_lt=(geopoly, D(m=SEARCH_WINDOW), 'spheroid'),
                                                         forest=1,
                                                         clouds=0,
                                                         )
            forest = flags_forest[idx]
            cloud = flags_clouds[idx]
            if forest == 1 and cloud == 0 and intersecting_polys.count() > 0:
                polys = [poly for poly in intersecting_polys]
                areas = [poly.mpoly.intersection(geopoly).area for poly in polys]
                dates = [poly.image_date_previous for poly in polys]
                max_intersection_area = np.argmax(areas)
                # Union of intersecting forest polygons:
                for poly in polys:
                    geopoly = geopoly.union(poly.mpoly)
                detection_date_union = [task.image_date_0, min(dates)]
                save_clearcut(geopoly,
                              avg_area,
                              detection_date_union,
                              forest,
                              cloud,
                              area_geodataframe[idx],
                              zone=polys[max_intersection_area].zone,
                              tile_index=task.tile_index,
                              )
            else:
                save_clearcut(geopoly,
                              avg_area,
                              detection_date,
                              forest,
                              cloud,
                              area_geodataframe[idx],
                              create_new_zone=True,
                              tile_index=task.tile_index,
                              )

    logger.info(f'---{time.time() - start_time} seconds --- for saving task_id: {task_id}')
