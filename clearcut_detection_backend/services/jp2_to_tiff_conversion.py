import logging
import os
import subprocess
import rasterio
import time
from clearcuts.models import TileInformation, Tile
from django.conf import settings
from downloader.models import SourceJp2Images
from concurrent.futures import ProcessPoolExecutor

logger = logging.getLogger('jp2_to_tiff_conversion')
tiff_dir = settings.MAPBOX_TIFFS_DIR
tiff_dir.mkdir(parents=True, exist_ok=True)


class Converter:
    """
    Convert jp2 images
    """
    def __init__(self, tile_index):
        self.tile_index = tile_index
        self.tile = Tile.objects.get(tile_index=self.tile_index)
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = f'{settings.BASE_DIR}/key.json'

    def convert_all_unconverted_to_tif(self):
        start_time = time.time()
        with ProcessPoolExecutor(max_workers=int(settings.MAX_WORKERS/2)) as executor:
            unconverted_list = SourceJp2Images.objects.prefetch_related('tile').filter(
                is_converted=0,
                tile__tile_index=self.tile_index
            )
            for uconv in unconverted_list:
                logger.info(f'tile_index: {self.tile_index}, image_date: {uconv.image_date}')
                output_filename = self.get_output_filename(self.tile_index, uconv.image_date)

                if not output_filename.is_file():
                    executor.submit(self.convert_image, uconv, output_filename)

        logger.info(f'{time.time() - start_time} seconds for for converting to tiff images for all')

    @staticmethod
    def get_output_filename(tile_index, tile_date):
        parent = tiff_dir / tile_index / str(tile_date)
        parent.mkdir(parents=True, exist_ok=True)

        return parent / f'{tile_index}.tif'

    def convert_image(self, unconverted, output_file, format='GTiff'):
        file_suffix = self.get_suffix_from_driver(format)
        if file_suffix:
            logger.info(f'file_suffix: {file_suffix}')
            with rasterio.open(unconverted.source_tci_location) as src:
                profile = src.profile
                profile['driver'] = format
                raster = src.read()


                crs = str(src.crs)
                self.tile.crs = crs
                self.tile.save()
                logger.info(f'crs: {crs}')

                kwargs = src.meta.copy()
                kwargs.update({
                    'driver': format
                })

                filename = output_file.with_suffix(file_suffix)
                logger.info(f'filename: {filename}')

                with rasterio.open(str(filename), 'w', **kwargs) as dst:
                    logger.info(f'dst: {dst}')
                    dst.write(raster)


            unconverted.is_converted = 1
            unconverted.save()

    def get_suffix_from_driver(self, driver):
        if driver == 'GTiff':
            return '.tif'
        if driver == 'JPEG':
            return '.jpeg'
        if driver == 'PNG':
            return '.png'

        logger.info(f'Unknown driver: {driver}')
        return



def jp2_to_tiff(tile_info_id=None):
    """
    Conversion raw satellite jp2 images to tiffs for mapbox
    """
    if tile_info_id:
        jp2files = list(TileInformation.objects.filter(id=tile_info_id))
    else:
        jp2files = list(TileInformation.objects
                        .filter(tile_index__is_tracked=1)
                        .filter(is_prepared=1)
                        .filter(is_converted=0)
                        .filter(source_tci_location__contains='jp2')
                        .filter(source_tci_location__contains='TCI')
                        .values_list('source_tci_location', flat=True))
    for file in jp2files:
        filename = os.path.basename(file).split('.')[0]
        logger.info('Converting %s to TIFF format', file)
        geo_tiff_file = os.path.join(settings.MAPBOX_TIFFS_DIR, f'{filename}.tiff')
        command_jp2_to_tiff = f'gdalwarp -of GTiff -overwrite -ot Byte -t_srs EPSG:4326 ' \
                              f'-wm 4096 -multi -wo NUM_THREADS=12 ' \
                              f'-co COMPRESS=DEFLATE -co PREDICTOR=2 {file} {geo_tiff_file}'

        # TODO: compressing and renaming dataset for backup reasons
        """
        rio calc "(asarray (take a 1) (take a 2) (take a 3))"
        --co compress=lzw --co tiled=true --co blockxsize=256 --co blockysize=256
        --name a={src_tile} {dst_tile}
        """
        # removing empty data pixels from dataset
        command_cutoff_nodata = f'rio edit-info --nodata 0 {geo_tiff_file}'

        for command in [command_jp2_to_tiff, command_cutoff_nodata]:
            try:
                result = process_command(command)
                if result:
                    tile_info = TileInformation.objects.get(source_tci_location=file)
                    tile_info.tile_location = geo_tiff_file
                    tile_info.is_converted = 1
                    tile_info.save()
            except (IOError, ValueError, FileNotFoundError, FileExistsError, Exception):
                logger.error('Error\n\n', exc_info=True)


def process_command(command):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
    stdout = process.communicate()[0].decode('utf8')
    # print(stdout.split())
    # TODO: Handle result
    return True
