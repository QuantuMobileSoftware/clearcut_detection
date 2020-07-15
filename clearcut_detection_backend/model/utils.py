"""
Model's helpers
"""
import io
import os.path
import logging

from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2 import service_account

import geopandas as gpd

logging.basicConfig(format='%(asctime)s %(message)s')

SCOPES = ['https://www.googleapis.com/auth/drive.file']

LANDCOVER_POLYGONS_PATH = 'data/landcover'
SENTINEL_TILES = f"{LANDCOVER_POLYGONS_PATH}/S2A_OPER_GIP_TILPAR_MPC__20151209T095117_V20150622T000000_21000101T000000_B00.kml"
LANDCOVER_GEOJSON = f'{LANDCOVER_POLYGONS_PATH}/landcover_polygons.geojson'

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
        polygon_path = os.path.join(LANDCOVER_POLYGONS_PATH, f"{self.tile}.geojson")
        if os.path.exists(polygon_path):
            logging.info(f'{self.tile} forest polygons file exists.')
            polygons = gpd.read_file(polygon_path)
        else:
            logging.info(f'{self.tile} forest polygons file does not exist. Creating polygons...')
            polygons = self.create_polygon()

        if len(polygons) > 0:
            polygons = polygons.to_crs(self.crs)
            polygons = list(polygons['geometry'])
        return polygons
    
    def create_polygon(self):
        polygons = []
        if os.path.exists(SENTINEL_TILES):
            sentinel_tiles = gpd.read_file(SENTINEL_TILES, driver='KML')
            sentinel_tiles = sentinel_tiles[sentinel_tiles['Name'] == self.tile]
            bounding_polygon = sentinel_tiles['geometry'].values[0]
            polygons = gpd.read_file(LANDCOVER_GEOJSON)
            polygons = polygons[polygons['geometry'].centroid.within(bounding_polygon)]
            
            polygon_path = os.path.join(LANDCOVER_POLYGONS_PATH, f"{self.tile}.geojson")
            polygons.to_file(polygon_path, driver='GeoJSON')
        return polygons


def weights_exists_or_download(path, file_id):
    if not os.path.exists(path):
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
