import os
import logging
import time
from osgeo import gdal
from concurrent.futures import as_completed, ThreadPoolExecutor, ProcessPoolExecutor
from pathlib import Path
from distutils.util import strtobool
from django.conf import settings
from django.core.exceptions import ObjectDoesNotExist
from clearcuts.geojson_save import save_from_task
from clearcuts.models import TileInformation, RunUpdateTask, Tile
from downloader.models import SourceJp2Images
from tiff_prepare.prepare_tif import to_tiff, get_ndvi, scale_img, merge_img_extra
from tiff_prepare.models import Prepared
from clearcut_detection_backend import app

prepare_tif = strtobool(os.environ.get('PREPARE_TIF', 'true'))
make_predict = strtobool(os.environ.get('MAKE_PREDICT', 'true'))

if not prepare_tif:
    convert_to_tiff = 0
    create_ndvi = 0
    scaling = 0
    merge = 0
else:
    convert_to_tiff = 1
    create_ndvi = 1
    create_ndmi = 1
    scaling = 1
    merge = 1


logger = logging.getLogger('model_call')

if settings.PATH_TYPE == 'fs':
    path_type = 0
else:
    logger.error(f'Unsupported file path in settings.PATH_TYPE: {settings.PATH_TYPE}')
    raise ValueError


class ModelCaller:
    """
    Class for asynchronous  calling of model's API
    """
    def __init__(self, tile_index):
        self.data_dir = settings.DATA_DIR
        self.tile_index = tile_index

    def start(self):
        tile_source_jp2_images_all = SourceJp2Images.objects.select_related('tile').filter(
            tile__tile_index=self.tile_index,
            source_tci_location__isnull=False,
            source_b04_location__isnull=False,
            source_b08_location__isnull=False,
            source_b8a_location__isnull=False,
            source_b11_location__isnull=False,
            source_b12_location__isnull=False,
            source_clouds_location__isnull=False,
            is_downloaded=7,
        ).order_by('image_date')

        if convert_to_tiff:
            self.convert_all_to_tiff(tile_source_jp2_images_all)
        if create_ndvi:
            self.create_all_ndvi(tile_source_jp2_images_all)
        if create_ndmi:
            self.create_all_ndmi(tile_source_jp2_images_all)
        if scaling:
            self.scale_all_bands(tile_source_jp2_images_all)
        if merge:
            self.merge_all_bands(tile_source_jp2_images_all)

    def convert_all_to_tiff(self, tile_source_jp2_images_all):
        future_to_tiff_list = []
        start_time = time.time()
        with ProcessPoolExecutor(max_workers=int(settings.MAX_WORKERS/2)) as executor:
            for tile_by_date in tile_source_jp2_images_all:
                tiff_dir = self.create_tiff_path(tile_by_date)  # create path for tiff images
                output_tiffs = self.create_output_tiffs(tiff_dir)  # defining temporary files names
                prepared, created = Prepared.objects.get_or_create(
                    tile=tile_by_date.tile,
                    image_date=tile_by_date.image_date
                )
                if not created:
                    prepared.success = 0
                    prepared.save()

                for band_to_tiff in self.all_bands_to_tif(tile_by_date, output_tiffs):
                    # here run executor
                    future = executor.submit(to_tiff, *band_to_tiff, prepared=prepared)
                    future_to_tiff_list.append(future)

        # for future in as_completed(future_to_tiff_list):
        #     logger.info(future.result())
        logger.info(f'{time.time() - start_time} seconds for for converting to tiff images for {self.tile_index}' )
        logger.info(f'converting all bands to *tif for {self.tile_index} finished')
        return

    def create_all_ndvi(self, tile_source_jp2_images_all):
        logger.info(f'creating ndvi band for {self.tile_index} started')
        with ProcessPoolExecutor(max_workers=int(settings.MAX_WORKERS/2)) as executor:
            future_to_ndvi_list = []
            start_time = time.time()
            for tile_by_date in tile_source_jp2_images_all:
                tiff_dir = self.create_tiff_path(tile_by_date)  # create path for tiff images

                output_tiffs = self.create_output_tiffs(tiff_dir)  # defining temporary files names

                prepared = Prepared.objects.get(tile=tile_by_date.tile, image_date=tile_by_date.image_date)
                if prepared != -1:
                    # here run executor
                    future = executor.submit(
                        get_ndvi,
                        output_tiffs.get('tiff_b4_name'),
                        output_tiffs.get('tiff_b8_name'),
                        output_tiffs.get('tiff_ndvi_name'),
                        prepared=prepared
                    )
                    future_to_ndvi_list.append(future)

        # for future in as_completed(future_to_ndvi_list):
        #     logger.info(future.result())
        logger.info(f'{time.time() - start_time} seconds for for creating ndvi band for {self.tile_index}')
        logger.info(f'creating ndvi band for {self.tile_index} finished')
        return

    def create_all_ndmi(self, tile_source_jp2_images_all):
        logger.info(f'creating ndmi band for {self.tile_index} started')
        future_to_ndmi_list = []
        start_time = time.time()
        with ProcessPoolExecutor(max_workers=int(settings.MAX_WORKERS / 2)) as executor:
            for tile_by_date in tile_source_jp2_images_all:
                tiff_dir = self.create_tiff_path(tile_by_date)  # create path for tiff images
                output_tiffs = self.create_output_tiffs(tiff_dir)  # defining temporary files names
                prepared = Prepared.objects.get(tile=tile_by_date.tile, image_date=tile_by_date.image_date)
                if prepared != -1:
                    # here run executor
                    future = executor.submit(
                        get_ndvi,
                        output_tiffs.get('tiff_b11_name'),
                        output_tiffs.get('tiff_b8a_name'),
                        output_tiffs.get('tiff_ndmi_name'),
                        prepared=prepared
                    )
                    future_to_ndmi_list.append(future)

            # for future in as_completed(future_to_ndmi_list):
            #     logger.info(future.result())
        logger.info(f'{time.time() - start_time} seconds for for creating ndmi band for {self.tile_index}')
        logger.info(f'creating ndmi band for {self.tile_index} finished')
        return

    def scale_all_bands(self, tile_source_jp2_images_all):
        logger.info(f'scaling all bands for {self.tile_index} started')
        future_scale_all_bands_list = []
        start_time = time.time()
        with ProcessPoolExecutor(max_workers=int(settings.MAX_WORKERS / 2)) as executor:
            for tile_by_date in tile_source_jp2_images_all:
                tiff_dir = self.create_tiff_path(tile_by_date)  # create path for tiff images
                output_tiffs = self.create_output_tiffs(tiff_dir)  # defining temporary files names
                prepared = Prepared.objects.get(tile=tile_by_date.tile, image_date=tile_by_date.image_date)
                if prepared != -1:
                    for band_to_scale in self.all_bands_to_scale(tile_by_date, output_tiffs):
                        # here run executor
                        future = executor.submit(scale_img, *band_to_scale, prepared=prepared)
                        future_scale_all_bands_list.append(future)

        # for future in as_completed(future_scale_all_bands_list):
        #     logger.info(future.result())
        logger.info(f'{time.time() - start_time} seconds for for scaling all bands for {self.tile_index}')
        logger.info(f'scaling all bands for {self.tile_index} finished')
        return
        
    def merge_all_bands(self, tile_source_jp2_images_all):
        logger.info(f'merge all bands for {self.tile_index} started')
        future_merge_all_bands_list = []
        start_time = time.time()
        with ProcessPoolExecutor(max_workers=int(settings.MAX_WORKERS/2)) as executor:
            for tile_by_date in tile_source_jp2_images_all:
                tiff_dir = self.create_tiff_path(tile_by_date)  # create path for tiff images
                output_tiffs = self.create_output_tiffs(tiff_dir)  # defining temporary files names
                tiff_output_name = tiff_dir / f'{tile_by_date.tile.tile_index}_{tile_by_date.image_date}_output.tif'
                prepared = Prepared.objects.get(tile=tile_by_date.tile, image_date=tile_by_date.image_date)
                if prepared != -1:
                    future = executor.submit(merge_img_extra,
                                             str(tiff_output_name),
                                             f"{output_tiffs.get('tiff_rgb_name')}",
                                             f"{output_tiffs.get('scaled_b8_name')}_scaled.tif",
                                             f"{output_tiffs.get('scaled_b8a_name')}_scaled.tif",
                                             f"{output_tiffs.get('scaled_b11_name')}_scaled.tif",
                                             f"{output_tiffs.get('scaled_b12_name')}_scaled.tif",
                                             f"{output_tiffs.get('scaled_ndvi_name')}_scaled.tif",
                                             f"{output_tiffs.get('scaled_ndmi_name')}_scaled.tif",
                                             is_verbose=0,
                                             is_quiet=0,
                                             separate=1,
                                             frmt='GTiff',
                                             prepared=prepared
                                             )
                    future_merge_all_bands_list.append(future)

        for future in as_completed(future_merge_all_bands_list):
            if future.result():
                self.remove_temp_files(future.result())
        logger.info(f'{time.time() - start_time} seconds for for merge all bands for {self.tile_index}')
        logger.info(f'merge all bands for {self.tile_index} finished')
        return

    @staticmethod
    def all_bands_to_tif(tile_by_date, output_tiffs):
        return (tile_by_date.source_b04_location, output_tiffs.get('tiff_b4_name')), \
               (tile_by_date.source_b08_location, output_tiffs.get('tiff_b8_name')),\
               (tile_by_date.source_b8a_location, output_tiffs.get('tiff_b8a_name')),\
               (tile_by_date.source_b11_location, output_tiffs.get('tiff_b11_name')),\
               (tile_by_date.source_b12_location, output_tiffs.get('tiff_b12_name')), \
               (tile_by_date.source_clouds_location, output_tiffs.get('tiff_clouds_name')),\
               (tile_by_date.source_tci_location, output_tiffs.get('tiff_rgb_name'), gdal.GDT_Byte)

    @staticmethod
    def all_bands_to_scale(tile_by_date, output_tiffs):
        return (output_tiffs.get('tiff_ndvi_name'), output_tiffs.get('scaled_ndvi_name')), \
               (output_tiffs.get('tiff_ndmi_name'), output_tiffs.get('scaled_ndmi_name')), \
               (tile_by_date.source_b08_location, output_tiffs.get('scaled_b8_name')), \
               (tile_by_date.source_b8a_location, output_tiffs.get('scaled_b8a_name')), \
               (tile_by_date.source_b11_location, output_tiffs.get('scaled_b11_name')), \
               (tile_by_date.source_b12_location, output_tiffs.get('scaled_b12_name'))

    @staticmethod
    def create_tiff_path(tile_by_date):
        """
        create path for tiff images
        """
        tiff_dir = settings.MODEL_TIFFS_DIR / tile_by_date.tile.tile_index / str(tile_by_date.image_date)
        tiff_dir.mkdir(parents=True, exist_ok=True)
        return tiff_dir

    @staticmethod
    def create_output_tiffs(tiff_dir):
        """
        defining temporary files names
        """
        return {'tiff_b4_name': tiff_dir / '_B04.tif',
                'tiff_b8_name': tiff_dir / '_B08.tif',
                'tiff_b8a_name': tiff_dir / '_B8A.tif',
                'tiff_b11_name': tiff_dir / '_B11.tif',
                'tiff_b12_name': tiff_dir / '_B12.tif',
                'tiff_rgb_name': tiff_dir / '_TCI.tif',
                'tiff_ndvi_name': tiff_dir / '_ndvi.tif',
                'tiff_ndmi_name': tiff_dir / '_ndmi.tif',
                'tiff_clouds_name': tiff_dir / 'clouds.tif',

                'scaled_b8_name': tiff_dir / '_scaled_B08.tif',
                'scaled_b8a_name': tiff_dir / '_scaled_B8A.tif',
                'scaled_b11_name': tiff_dir / '_scaled_B11.tif',
                'scaled_b12_name': tiff_dir / '_scaled_B12.tif',
                'scaled_ndvi_name': tiff_dir / '_scaled_ndvi.tif',
                'scaled_ndmi_name': tiff_dir / '_scaled_ndmi.tif',
                }

    @staticmethod
    def predict_and_save_from_result(tile_index):
        tile_list = TileInformation.objects.filter(
            tile_index__tile_index=tile_index,
            is_prepared=1
        ).order_by('tile_name')
        if len(tile_list) < 2:
            logger.error(f'cant predict tile {tile_index}, len(tiles) < 2')
            return
        path_img_0 = tile_list[0].model_tiff_location
        path_img_1 = tile_list[1].model_tiff_location
        image_date_0 = tile_list[0].tile_date
        image_date_1 = tile_list[1].tile_date

        path_clouds_0 = str(Path(path_img_0).parent / 'clouds.tiff')
        path_clouds_1 = str(Path(path_img_1).parent / 'clouds.tiff')

        tile = Tile.objects.get(tile_index=tile_index)  # TODO

        task = RunUpdateTask(tile_index=tile,
                             path_type=path_type,
                             path_img_0=path_img_0,
                             path_img_1=path_img_1,
                             image_date_0=image_date_0,
                             image_date_1=image_date_1,
                             path_clouds_0=path_clouds_0,
                             path_clouds_1=path_clouds_1
                             )
        task.save()

        if make_predict:
            logger.info(f'send task_id: {task.id} to queue')
            result = model_add_task(task.id)
            res = result.get()
            try:
                save_from_task(res)
                return f'task_id: {res} done'
            except (ValueError, Exception):
                logger.error(f'cant do save_from_task({res})')
        else:
            logger.info(f'skip predict, save task_id: {task.id}')
            save_from_task(task.id)

    @staticmethod
    def remove_temp_files(path):
        path = Path(path).parent
        logger.info(f'Try remove temp files for {path}')
        temp_files = Path(path).glob(f'_*.tif')
        try:
            for file in temp_files:
                file.unlink()
            logger.info(f'temp files for {path} were removed')
        except OSError:
            logger.error('Error\n\n', exc_info=True)


def model_add_task(task_id):
    """
    Add run update task task in to 'run_update_task' queue
    :param task_id:
    :return:
    """
    result = app.send_task(
        name='tasks.run_model_predict',
        queue='model_predict_queue',
        kwargs={'task_id': task_id},
        task_track_started=True,
        )
    return result
