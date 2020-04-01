import boto3
import datetime
import json
import os
import requests
import sys
import threading

from boto3.s3.transfer import TransferConfig
from concurrent.futures import ThreadPoolExecutor
from django.conf import settings

from clearcuts.models import TileInformation

mapbox_username = settings.MAPBOX_USER
access_token = settings.MAPBOX_ACCESS_TOKEN
url_credentials = f'https://api.mapbox.com/uploads/v1/{mapbox_username}/credentials?access_token={access_token}'
url_upload = f'https://api.mapbox.com/uploads/v1/{mapbox_username}?access_token={access_token}'


def start_upload():
    """
    Get all tiles elegible for upload
    Start upload in threads
    :return:
    """
    executor = ThreadPoolExecutor(max_workers=10)
    tiles = TileInformation.objects\
        .filter(tile_location__contains='tiff')\
        .filter(tile_location__contains='TCI')
    for tile in tiles:
        executor.submit(upload_to_mapbox, tile)
    return executor


def upload_to_mapbox(tile):
    """
    Get temp creds for S3.
    Multipart upload tile
    Mapping mapbox info into the model
    :param tile:
    :return:
    """
    s3_creds = json.loads(requests.post(url_credentials).content)
    multi_part_upload_with_s3(s3_creds, tile.tile_location)
    file_name = tile.tile_location.rsplit('/')[-1].rsplit('.', 1)[0]
    payload = {
        "url": f"{s3_creds['url']}",
        "tileset": f"{mapbox_username}.{file_name}",
        "name": f"{file_name}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
    }
    mapbox_tile_info = json.loads(
        requests.post(
            url=url_upload,
            data=json.dumps(payload),
            headers={'Content-type': 'application/json', 'Accept': 'text/plain'}
        ).content
    )

    tile.mapbox_tile_id = mapbox_tile_info.get('id')
    tile.mapbox_tile_name = mapbox_tile_info.get('name')
    tile.mapbox_tile_layer = mapbox_tile_info.get('tileset')

    os.remove(tile.tile_location)
    tile.tile_location = None
    tile.save()

    return mapbox_tile_info


def multi_part_upload_with_s3(aws_creds, uploaded_file_path):
    """
    Creates s3 resource with temporary creds obtained from
        mapbox API /credentials
    Sets config for transferring big files.
    Uploads the data to S3 via config with completeness indication
    :return: None
    """

    s3 = boto3.resource(
        's3',
        aws_access_key_id=aws_creds.get('accessKeyId'),
        aws_secret_access_key=aws_creds.get('secretAccessKey'),
        aws_session_token=aws_creds.get('sessionToken'),
    )

    config = TransferConfig(multipart_threshold=1024 * 25,
                            max_concurrency=10,
                            multipart_chunksize=1024 * 25,
                            use_threads=True)

    s3.meta.client.upload_file(
        uploaded_file_path,
        aws_creds.get('bucket'),
        aws_creds.get('key'),
        Config=config,
        Callback=ProgressPercentage(uploaded_file_path))
    # TODO: create verification file on S3 and return status


"""
ProgressPercentage is useless if upload is done with threading 
"""


class ProgressPercentage(object):
    """
    Progressbar
    """

    def __init__(self, filename):
        self._filename = filename
        self._size = float(os.path.getsize(filename))
        self._uploaded = 0
        self._lock = threading.Lock()

    def __call__(self, bytes_amount):
        with self._lock:
            self._uploaded += bytes_amount
            percentage = (self._uploaded / self._size) * 100
            sys.stdout.write(
                "\r%s  %s / %s  (%.2f%%)" % (
                    self._filename, self._uploaded, self._size,
                    percentage))
            sys.stdout.flush()
