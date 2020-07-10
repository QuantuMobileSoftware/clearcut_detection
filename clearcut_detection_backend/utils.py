"""
Script helpers
"""
from enum import Enum
from zipfile import ZipFile
from io import BytesIO
import requests


class Bands(Enum):
    TCI = 'TCI'
    B04 = 'B04'
    B08 = 'B08'
    B8A = 'B8A'
    B11 = 'B11'
    B12 = 'B12'


def download_without_progress(url):
    """
    Download data from a URL, returning a BytesIO containing the loaded data.

    Parameters
    ----------
    url : str
        A URL that can be understood by ``requests.get``.

    Returns
    -------
    data : BytesIO
        A BytesIO containing the downloaded data.
    """
    resp = requests.get(url)
    resp.raise_for_status()
    return BytesIO(resp.content)


def fetch_all_from_zip(file, path_to_extract):
    """
    fetch all files from archive
    :param file: BytesIO or file path
    :param path_to_extract:
    :return:
    """
    with ZipFile(file) as zip_file:
        zip_file.extractall(path_to_extract)


def fetch_file_from_zip(file, source, destination):
    """
    fetch specific file from archive to destination path
    :param file: BytesIO or file path
    :param source: file name to be extracted
    :param destination: path for extraction
    :return:
    """
    with ZipFile(file) as zip_file:
        zip_file.extract(source, destination)
