import os
import logging
import time
import geopandas
from datetime import datetime, timedelta
from celery import group
from pathlib import Path
import rasterio
from rasterio.mask import mask
from rasterio.windows import Window
from rasterio import Affine
from rasterio.features import shapes
import geopandas as gpd
from fiona.crs import from_epsg
from shapely.geometry import box
import pycrs

from distutils.util import strtobool
from django.conf import settings
from django.http import JsonResponse, HttpResponse, Http404
from django.contrib.gis.gdal import GDALRaster
from clearcut_detection_backend import app
from clearcuts.models import RunUpdateTask, TileInformation, Tile, Clearcut
from tiff_prepare.models import Prepared
from downloader.models import SourceJp2Images
from clearcuts.geojson_save import save_from_task
from downloader.services import SentinelDownload

from django.contrib.gis.geos import Point, Polygon
import shapely.geometry


make_predict = strtobool(os.environ.get('MAKE_PREDICT', 'true'))
logger = logging.getLogger('create_update_task')

if settings.PATH_TYPE == 'fs':
    path_type = 0
else:
    logger.error(f'Unsupported file path in settings.PATH_TYPE: {settings.PATH_TYPE}')
    raise ValueError


class CreateUpdateTask:
    def __init__(self, tile_index):
        self.tile_index = tile_index

    @property
    def get_all_prepared(self):
        """
        get all records for prepared images
        """
        return Prepared.objects.filter(tile_id__tile_index=self.tile_index, success=1).order_by('image_date')

    @property
    def get_new_prepared(self):
        """
        get all new records for prepared images
        """
        return Prepared.objects.filter(tile_id__tile_index=self.tile_index, success=1, is_new=1)

    def run_from_prepared(self, prepared):
        """
        create tasks for update from prepared records
        :prepared: list of Prepared.objects
        """
        logger.info(f'len(prepared): {len(prepared)}')
        if len(prepared) < 2:
            logger.error(f'cant predict tile {self.tile_index}, len(prepared) < 2')
            return

        task_list = []
        for i in range(len(prepared) - 1):
            path_img_0 = prepared[i].model_tiff_location
            path_img_1 = prepared[i + 1].model_tiff_location
            image_date_0 = prepared[i].image_date
            image_date_1 = prepared[i + 1].image_date

            path_clouds_0 = prepared[i].cloud_tiff_location
            path_clouds_1 = prepared[i + 1].cloud_tiff_location

            tile = Tile.objects.get(tile_index=self.tile_index)

            task = RunUpdateTask(tile=tile,
                                 path_type=path_type,
                                 path_img_0=path_img_0,
                                 path_img_1=path_img_1,
                                 image_date_0=image_date_0,
                                 image_date_1=image_date_1,
                                 path_clouds_0=path_clouds_0,
                                 path_clouds_1=path_clouds_1
                                 )
            task.save()
            prepared[i].is_new = 0
            prepared[i].save()

            logger.info(f'send task_id: {task.id} to queue')
            task_list.append(app.send_task(
                name='tasks.run_model_predict',
                queue='model_predict_queue',
                kwargs={'task_id': task.id},
                ignore_result=False,
            ))

        if make_predict:
            job = group(task_list)
            task_saved = False
            while len(job.tasks):
                time.sleep(2 * 10)
                logger.info(f'len(job.tasks): {len(job.tasks)}')
                cnt = 0
                for j in job.tasks:
                    # logger.info(j)
                    # logger.info(j.successful())
                    if j.successful():
                        logger.info(j.result)
                        task_id = j.result
                        job.tasks.pop(cnt)
                        try:
                            save_from_task(task_id)
                            task_saved = True
                        except (ValueError, Exception):
                            logger.error(f'cant do save_from_task({task_id})', exc_info=True)
                        if task_saved:
                            self.update_tile_dates(task_id)
                            task_saved = False
                        break
                    cnt += 1

    @staticmethod
    def update_tile_dates(task_id):
        task = RunUpdateTask.objects.get(id=task_id)
        tile = Tile.objects.get(id=task.tile.id)
        if tile.first_date is None or tile.first_date > task.image_date_0:
            tile.first_date = task.image_date_0

        if tile.last_date is None or tile.last_date < task.image_date_1:
            tile.last_date = task.image_date_1
        tile.save()


class Preview:

    def get_or_create_preview(self, clearcut):
        preview_previous_path = Path(clearcut.preview_previous_path) if clearcut.preview_previous_path else None
        preview_current_path = Path(clearcut.preview_current_path) if clearcut.preview_current_path else None

        if not preview_previous_path:
            source_img_previous_path, source_img_current_path = self.get_or_download_tci_images(clearcut)
            preview_previous_path, preview_current_path = self.get_preview_from_local_images(
                source_img_previous_path,
                source_img_current_path,
                clearcut
            )
            clearcut.preview_previous_path = preview_previous_path
            clearcut.preview_current_path = preview_current_path
            clearcut.save()

        if not preview_previous_path.is_file() or not preview_current_path.is_file():
            source_img_previous_path, source_img_current_path = self.get_or_download_tci_images(clearcut)
            preview_previous_path, preview_current_path = self.get_preview_from_local_images(
                source_img_previous_path,
                source_img_current_path,
                clearcut
            )
        return preview_previous_path, preview_current_path

    def get_or_download_tci_images(self, clearcut):
        """
        Try to get record with unique combination of tile_index and image_date from downloader_sourcejp2images table
        If record not exists or if file in source_tci_location is not exists
        than download file from Google cloud storage.
        :param clearcut: Clearcuts obj
        """
        source_img_previous_path, source_img_current_path = self.get_info_from_db(clearcut)
        source_img_previous_path = Path(source_img_previous_path) if source_img_previous_path is not None else None
        source_img_current_path = Path(source_img_current_path) if source_img_current_path is not None else None

        if not source_img_previous_path:

            sentinel_downloader = SentinelDownload(clearcut.zone.tile.tile_index)
            source_img_previous_path = sentinel_downloader.download_tci_images_on_date_from_google_cloud_storage(
                clearcut.image_date_previous
            )

            self.set_source_tci_location_to_db(
                clearcut.zone.tile,
                clearcut.image_date_previous,
                source_img_previous_path)

        elif not source_img_previous_path.is_file():
            sentinel_downloader = SentinelDownload(clearcut.zone.tile.tile_index)
            source_img_previous_path = sentinel_downloader.download_tci_images_on_date_from_google_cloud_storage(
                clearcut.image_date_previous
            )

        if not source_img_current_path:
            sentinel_downloader = SentinelDownload(clearcut.zone.tile.tile_index)
            source_img_current_path = sentinel_downloader.download_tci_images_on_date_from_google_cloud_storage(
                clearcut.image_date_current
            )

            self.set_source_tci_location_to_db(
                clearcut.zone.tile,
                clearcut.image_date_current,
                source_img_current_path)

        elif not source_img_current_path.is_file():
            sentinel_downloader = SentinelDownload(clearcut.zone.tile.tile_index)
            source_img_current_path = sentinel_downloader.download_tci_images_on_date_from_google_cloud_storage(
                clearcut.image_date_current
            )

        return source_img_previous_path, source_img_current_path

    def get_preview_from_local_images(self, source_img_previous, source_img_current, clearcut):
        image_date_previous = clearcut.image_date_previous
        image_date_current = clearcut.image_date_current
        tile_index = clearcut.zone.tile.tile_index

        file_path_previous = settings.POLYGON_TIFFS_DIR / tile_index / str(image_date_previous)
        file_path_previous.mkdir(parents=True, exist_ok=True)
        preview_previous_path = file_path_previous / f'{clearcut.id}.tiff'

        file_path_current = settings.POLYGON_TIFFS_DIR / tile_index / str(image_date_current)
        file_path_current.mkdir(parents=True, exist_ok=True)
        preview_current_path = file_path_current / f'{clearcut.id}.tiff'

        polygon = clearcut.mpoly

        self.create_preview_from_local_image(source_img_previous, preview_previous_path, polygon)
        self.create_preview_from_local_image(source_img_current, preview_current_path, polygon)

        return preview_previous_path, preview_current_path

    @staticmethod
    def save_polygon_to_file(extent, crs, path):
        # x_min, y_min, x_max, y_max = polygon.extent
        geom = box(*extent)
        geo = gpd.GeoDataFrame({'geometry': geom}, index=[0], crs=crs.data)
        # geo = geo.to_crs(crs=crs.data)
        geo.to_file(path, driver='GeoJSON')

    def create_preview_from_local_image(self, source_img_path, preview_path, polygon):
        border = 100
        padding = border / 2
        with rasterio.open(source_img_path) as src:

            mpoly = polygon.transform(str(src.crs), clone=True)
            x_min, y_min, x_max, y_max = mpoly.extent
            polygon_path = Path(settings.POLYGON_TIFFS_DIR / 'polygon_orig_1.geojson')

            self.save_polygon_to_file((x_min, y_min, x_max, y_max), src.crs, polygon_path)

            affine = Affine(src.transform[0],
                            src.transform[1],
                            x_min,
                            src.transform[3],
                            src.transform[4],
                            y_max
                            )




            # row_max, col_min = rasterio.transform.rowcol(src.transform, x_min, y_min)
            row_min, col_max = rasterio.transform.rowcol(src.transform, x_max, y_max)
            col_max_1, row_min_1 = ~src.transform * (x_max, y_max)
            import math
            row_max, col_min = rasterio.transform.rowcol(src.transform, x_min, y_min, op=round, precision=6)
            col_min_1, row_max_1 = ~src.transform * (x_min, y_min)

            row_min = row_min + 1
            row_max = row_max + 1






            write_window = Window.from_slices([row_min, row_max, ], [col_min, col_max])

            kwargs = src.meta.copy()
            kwargs.update({
                'height': write_window.height,
                'width': write_window.width,
                # 'transform': rasterio.windows.transform(write_window, src.transform),
                "transform": affine,
                'driver': 'GTiff'}
            )

            parent_path = preview_path.parent
            name = f'{preview_path.stem}_orig.tiff'

            with rasterio.open(str(parent_path / name), 'w', **kwargs) as dst:
                image = src.read(window=write_window)
                dst.write(image)

                # shape = shapes(image, mask=None, transform=dst.transform)
                # s, v = shape.__next__()
                # geom = {'properties': {'raster_val': v}, 'geometry': s}
                # gpd_polygonized_raster = gpd.GeoDataFrame.from_features(geom, crs=dst.crs)
                # gpd_polygonized_raster.to_file(str(parent_path / 'poligon_from_image_1.geojson'), driver='GeoJSON')

                results = (
                    {'properties': {'raster_val': v}, 'geometry': s}
                    for i, (s, v) in enumerate(shapes(image, mask=None, transform=dst.transform)))

                geoms = list(results)

                gpd_polygonized_raster = gpd.GeoDataFrame.from_features(geoms, crs=dst.crs)
                x_min, y_min, x_max, y_max = gpd_polygonized_raster.total_bounds
                polygon_path = Path(settings.POLYGON_TIFFS_DIR / 'polygon_orig_2.geojson')
                self.save_polygon_to_file((x_min, y_min, x_max, y_max), src.crs, polygon_path)

                gpd_polygonized_raster.to_file(str(parent_path / 'poligon_from_image_2.geojson'), driver='GeoJSON')




            mpoly = mpoly.buffer_with_style(100, quadsegs=8, end_cap_style=2, join_style=1, mitre_limit=5.0)

            x_min, y_min, x_max, y_max = mpoly.extent
            polygon_path = Path(settings.POLYGON_TIFFS_DIR / 'polygon_buffered.geojson')
            self.save_polygon_to_file((x_min, y_min, x_max, y_max), src.crs, polygon_path)

            transform = src.transform

            affine = Affine(src.transform[0],
                            src.transform[1],
                            x_min,
                            src.transform[3],
                            src.transform[4],
                            y_max
                            )

            # row_max, col_min = rasterio.transform.rowcol(src.transform, x_min, y_min)
            # row_min, col_max = rasterio.transform.rowcol(src.transform, x_max, y_max)

            row_min, col_max = rasterio.transform.rowcol(src.transform, x_max, y_max)
            col_max_1, row_min_1 = ~src.transform * (x_max, y_max)
            import math
            row_max, col_min = rasterio.transform.rowcol(src.transform, x_min, y_min, op=round, precision=6)
            col_min_1, row_max_1 = ~src.transform * (x_min, y_min)

            row_min = row_min + 1
            row_max = row_max + 1

            write_window = Window.from_slices([row_min, row_max, ], [col_min, col_max])


            kwargs = src.meta.copy()
            kwargs.update({
                'height': write_window.height,
                'width': write_window.width,
                # 'transform': rasterio.windows.transform(write_window, src.transform),
                "transform": affine,
                'driver': 'GTiff'}
            )

            with rasterio.open(str(preview_path), 'w', **kwargs) as dst:
                dst.write(src.read(window=write_window))


        # logger.info(f'coords:{coords}')

    def create_preview_from_local_image_2(self, source_img_path, preview_path, polygon):
        border = 100
        padding = border / 2

        data = rasterio.open(source_img_path)
        x_min, y_min, x_max, y_max = polygon.extent

        # x_min = x_min - padding if x_min >= padding else 0
        # y_min = y_min - padding if y_min >= padding else 0
        # x_max = x_max + padding
        # y_max = y_max + padding



        geom = box(x_min, y_min, x_max, y_max)

        geo = gpd.GeoDataFrame({'geometry': geom}, index=[0], crs=from_epsg(4326))
        geo = geo.to_crs(crs=data.crs.data)
        p = Path(settings.POLYGON_TIFFS_DIR / 'aaaa.geojson')
        geo.to_file(p, driver='GeoJSON')


        if p.is_file():
            logger.info('rrrrrrrrrrrrrrrrrr')
        coords = self.get_features(geo)
        coords_2 = geo['geometry'].values[0]

        from shapely.geometry import mapping
        coords_3 = mapping(coords_2)

        out_img, out_transform = mask(data,
                                      shapes=[coords_3, ],
                                      crop=True,
                                      # all_touched=True,
                                      # filled=False,
                                      # pad=True,
                                      # pad_width=0
                                      )
        out_meta = data.meta.copy()
        logger.info(f'out_meta: {out_meta}')
        epsg_code = int(data.crs.data['init'][5:])
        logger.info(f'epsg_code: {epsg_code}')



        out_meta.update({"driver": "GTiff",
                         "height": out_img.shape[1],
                         "width": out_img.shape[2],
                         "transform": out_transform,
                         # "transform": rasterio.windows.transform(out_img, data.transform),

                         "crs": pycrs.parse.from_epsg_code(epsg_code).to_proj4()
                         }
                        )
        with rasterio.open(preview_path, "w", **out_meta) as dest:
            dest.write(out_img)

    @staticmethod
    def get_features(gdf):
        """Function to parse features from GeoDataFrame in such a manner that rasterio wants them"""
        import json
        return [json.loads(gdf.to_json())['features'][0]['geometry']]

    @staticmethod
    def get_info_from_db(clearcut):
        source_img_previous = None
        source_img_current = None
        try:
            source_img_previous = SourceJp2Images.objects.get(
                tile=clearcut.zone.tile,
                image_date=clearcut.image_date_previous)
        except SourceJp2Images.DoesNotExist:
            logger.info(f'No record for {clearcut.zone.tile.tile_index} - {clearcut.image_date_previous}')

        try:
            source_img_current = SourceJp2Images.objects.get(
                tile=clearcut.zone.tile,
                image_date=clearcut.image_date_current,)
        except SourceJp2Images.DoesNotExist:
            logger.info(f'No record for {clearcut.zone.tile.tile_index} - {clearcut.image_date_current}')

        return source_img_previous.source_tci_location, source_img_current.source_tci_location

    @staticmethod
    def set_source_tci_location_to_db(tile, image_date, source_tci_location):
        source_img_current, created = SourceJp2Images.objects.get_or_create(
            tile=tile,
            image_date=image_date, 
        )
        if created:
            source_img_current.source_tci_location = source_tci_location
            source_img_current.save()
