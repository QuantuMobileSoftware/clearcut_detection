"""
Model's helpers
"""
import io
import os.path
import logging
from pathlib import Path
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2 import service_account
import geopandas as gpd
from config import SCOPES, LANDCOVER_POLYGONS_PATH, SENTINEL_TILES, LANDCOVER_GEOJSON

logging.basicConfig(format='%(asctime)s %(message)s')


class LandcoverPolygons:
    """
    LandcoverPolygon class to access forest polygons. Before usage,
    be sure that SENTINEL_TILES file is downloaded.
    SENTINEL_TILES_POLYGONS = 'https://sentinel.esa.int/documents/247904/1955685/S2A_OPER_GIP_TILPAR_MPC__20151209T095117_V20150622T000000_21000101T000000_B00.kml'
    
    :param tile: tile name (str), e.g. '36UYA'
    :param crs: coordinate system (str), e.g. 'EPSG:4326'

    :return polygons: list of forest polygons within a tile in CRS of a S2A image
    """

    def __init__(self, tile, crs):
        self.tile = tile
        self.crs = crs
        gpd.io.file.fiona.drvsupport.supported_drivers['KML'] = 'rw'

    def get_polygon(self):
        polygon_path = LANDCOVER_POLYGONS_PATH / f'{self.tile}.geojson'
        logging.info(f'LANDCOVER_POLYGONS_PATH: {polygon_path}')
        if polygon_path.is_file():
            logging.info(f'{self.tile} forests polygons file exists.')
            polygons = gpd.read_file(polygon_path)
        else:
            logging.info(f'{self.tile} forests polygons file does not exist. Creating polygons...')
            polygons = self.create_polygon()

        if len(polygons) > 0:
            polygons = polygons.to_crs(self.crs)
            polygons = list(polygons['geometry'])
        else:
            logging.info('No forests polygons.')
        return polygons
    
    def create_polygon(self):
        polygons = []
        if SENTINEL_TILES.is_file():
            logging.info(f'read forests_polygons_file: {SENTINEL_TILES}, for tile {self.tile}')
            sentinel_tiles = gpd.read_file(SENTINEL_TILES, driver='KML')
            sentinel_tiles = sentinel_tiles[sentinel_tiles['Name'] == self.tile]
            logging.info(f'sentinel_tiles for {self.tile}: {sentinel_tiles}')
            bounding_polygon = sentinel_tiles['geometry'].values[0]
            polygons = gpd.read_file(LANDCOVER_GEOJSON)
            polygons = polygons[polygons['geometry'].intersects(bounding_polygon)]
            polygon_path = LANDCOVER_POLYGONS_PATH / f'{self.tile}.geojson'
            logging.info(f'forests_polygons_file_path: {polygon_path}')
            polygons.to_file(polygon_path, driver='GeoJSON')
        else:
            logging.error(f'{SENTINEL_TILES} doth not exists')
            raise FileNotFoundError
        return polygons


def weights_exists_or_download(path, file_id):
    if not Path(path).exists():
        creds_file = os.environ.get('CREDENTIAL_FILE')
        creds = service_account.Credentials.from_service_account_file(creds_file, scopes=SCOPES)

        service = build('drive', 'v3', credentials=creds)
        request = service.files().get_media(fileId=file_id)

        fh = io.FileIO('unet_v4.pth', mode='wb')
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
            print(f'Download {int(status.progress() * 100)}')

    return path
