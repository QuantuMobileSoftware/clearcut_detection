import os
import logging
import time
import geopandas as gp
import numpy as np
from django.conf import settings
from django.contrib.gis.geos import GEOSGeometry
from django.contrib.gis.measure import D
from .models import Clearcut, Zone, RunUpdateTask, NotClearcut

import rasterio
from rasterio.windows import Window
from rasterio import Affine
from services.jp2_to_tiff_conversion import Converter


SEARCH_WINDOW = 50

logger = logging.getLogger('update')


class CreatePreview:
    def __init__(self, task):
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = f'{settings.BASE_DIR}/key.json'
        self.tile = task.tile
        self.tile_index = self.tile.tile_index
        self.crs = self.tile.crs
        self.image_date_previous = task.image_date_0
        self.image_date_current = task.image_date_1

        self.source_img_previous = Converter.get_output_filename(self.tile_index, self.image_date_previous)
        self.source_img_current = Converter.get_output_filename(self.tile_index, self.image_date_current)

        self.src_previous = rasterio.open(self.source_img_previous)
        self.src_current = rasterio.open(self.source_img_current)

    def srs_close(self):
        self.src_previous.close()
        self.src_current.close()

    def create_previews_for_clearcut(self, clearcut):
        polygon = clearcut.mpoly
        buffered_polygon = self.create_buffered_polygon(polygon)
        preview_previous_path, preview_current_path = self.create_preview_path_for_clearcut(clearcut)
        self.create_preview_from_src(self.src_previous, preview_previous_path, buffered_polygon)
        self.create_preview_from_src(self.src_current, preview_current_path, buffered_polygon)
        return preview_previous_path, preview_current_path

    @staticmethod
    def create_preview_from_src(src, preview_path, polygon):
        x_min, y_min, x_max, y_max = polygon.extent
        affine = Affine(src.transform[0],
                        src.transform[1],
                        x_min,
                        src.transform[3],
                        src.transform[4],
                        y_max
                        )

        row_min, col_max = rasterio.transform.rowcol(src.transform, x_max, y_max)
        row_max, col_min = rasterio.transform.rowcol(src.transform, x_min, y_min, op=round, precision=6)
        row_min = row_min + 1
        row_max = row_max + 1
        write_window = Window.from_slices([row_min, row_max, ], [col_min, col_max])
        raster = src.read(window=write_window)
        kwargs = src.meta.copy()
        kwargs.update({
            'height': write_window.height,
            'width': write_window.width,
            "transform": affine,
            'driver': 'GTiff'
        })

        with rasterio.open(str(preview_path), 'w', **kwargs) as dst:
            dst.write(raster)

        kwargs.update({'driver': 'PNG'})
        with rasterio.open(preview_path.with_suffix('.png'), 'w', **kwargs) as dst:
            dst.write(raster)

    def create_preview_path_for_clearcut(self, clearcut):
        file_path_previous = settings.POLYGON_TIFFS_DIR / self.tile_index / str(self.image_date_previous)
        file_path_previous.mkdir(parents=True, exist_ok=True)
        preview_previous_path = file_path_previous / f'{clearcut.id}.{settings.POLYGON_FORMAT}'

        file_path_current = settings.POLYGON_TIFFS_DIR / self.tile_index / str(self.image_date_current)
        file_path_current.mkdir(parents=True, exist_ok=True)
        preview_current_path = file_path_current / f'{clearcut.id}.{settings.POLYGON_FORMAT}'

        return preview_previous_path, preview_current_path

    def create_buffered_polygon(self, polygon):
        mpoly = polygon.transform(self.crs, clone=True)
        mpoly = mpoly.buffer_with_style(settings.POLYGON_BUFFER, quadsegs=8, end_cap_style=2, join_style=1,
                                        mitre_limit=5.0)
        return mpoly


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
        preview=None,
        ):
    if area_in_meters > avg_area * area_threshold and poly.geom_type == 'Polygon':
        if create_new_zone:
            zone = Zone()
            zone.tile = tile
            zone.save()
        clearcut = Clearcut(
            image_date_previous=detection_date[0],
            image_date_current=detection_date[1],
            mpoly=poly,
            area=area_in_meters,
            zone=zone,
            forest=forest,
            clouds=cloud,
            centroid=poly.centroid,
        )
        clearcut.save()
        preview_previous_path, preview_current_path = preview.create_previews_for_clearcut(clearcut)
        clearcut.preview_previous_path = preview_previous_path
        clearcut.preview_current_path = preview_current_path
        clearcut.save()
        logger.info(f'clearcut saved with id={clearcut.id}')


def save_from_task(task_id):
    logger.info(f'task_id: {task_id}')
    start_time = time.time()
    task = RunUpdateTask.objects.prefetch_related('tile').get(id=task_id)
    detection_date = [task.image_date_0, task.image_date_1]
    logger.info(f'detection_date: {detection_date}')

    preview = CreatePreview(task)

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
            # zone__tile_id=task.tile_id,
            mpoly__dwithin=(geopoly, D(m=SEARCH_WINDOW), 'spheroid'),
            ))
        if len(intersecting_polys) == 0:
            save_clearcut(geopoly,
                          avg_area,  # TODO
                          detection_date,
                          forest,
                          cloud,
                          area_geodataframe[idx],
                          create_new_zone=True,
                          tile=task.tile,
                          preview=preview,
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
                          preview=preview,
                          )
    preview.srs_close()
    logger.info(f'---{time.time() - start_time} seconds --- for saving task_id: {task_id}')
