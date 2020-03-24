import logging
import os
import subprocess

from clearcuts.models import TileInformation
from utils import path_exists_or_create

logging.basicConfig(format='%(asctime)s %(message)s')
MAPBOX_TIFFS_DIR = path_exists_or_create('data/mapbox_tiffs')


def jp2_to_tiff():
    """
    Conversion raw satellite jp2 images to tiffs for mapbox
    """
    jp2files = list(TileInformation.objects
                    .filter(source_tci_location__contains='jp2')
                    .filter(source_tci_location__contains='TCI')
                    .values_list('source_tci_location', flat=True))
    for file in jp2files:
        filename = os.path.basename(file).split('.')[0]
        logging.warning('Converting %s to TIFF format', file)
        geo_tiff_file = os.path.join(MAPBOX_TIFFS_DIR, f'{filename}.tiff')
        command_jp2_to_tiff = f'gdalwarp -of GTiff -overwrite -ot Byte -t_srs EPSG:4326 ' \
                              f'-wm 4096 -multi -wo NUM_THREADS=ALL_CPUS ' \
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
            result = process_command(command)
        if result:
            tile_info = TileInformation.objects.get(source_tci_location=file)
            tile_info.tile_location = geo_tiff_file
            tile_info.save()


def process_command(command):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
    stdout = process.communicate()[0].decode('utf8')
    # print(stdout.split())
    # TODO: Handle result
    return True
