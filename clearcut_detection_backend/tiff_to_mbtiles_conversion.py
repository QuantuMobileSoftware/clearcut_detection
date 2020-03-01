import logging
import subprocess

from clearcuts.models import TileInformation

logging.basicConfig(format='%(asctime)s %(message)s')
DATA_DIR = 'data'

"""
until source data is obtained from Sentinel no need to use this conversion 
Mbtiles are much more heavier and image quality obtained from Sentinel is not worth it.
"""


def tiff_to_mbtiles():

    tiff_files = list(TileInformation.objects
                      .filter(tile_location__contains='tiff')
                      .filter(tile_location__contains='TCI')
                      .values_list('tile_location', flat=True))

    for tiff in tiff_files:
        temp_tiff = tiff+'.3'
        mbtiles = f'{tiff.split(".")[0]}.mbtiles'
        cmd1 = f"rio stack {tiff} --bidx 1,2,3 --overwrite {temp_tiff}"
        cmd2 = f"rio mbtiles {temp_tiff} -o {mbtiles} -j 4 -f PNG --overwrite --zoom-levels 14..17"
        logging.info("running {}".format(cmd1))
        subprocess.run(cmd1.split())
        logging.info("running {}".format(cmd2))
        subprocess.run(cmd2.split())
