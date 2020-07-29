import logging
import os
import subprocess

from clearcuts.models import TileInformation
from django.conf import settings

logger = logging.getLogger('jp2_to_tiff_conversion')
settings.MAPBOX_TIFFS_DIR.mkdir(parents=True, exist_ok=True)


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
