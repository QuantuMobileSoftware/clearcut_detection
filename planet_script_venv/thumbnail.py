import os
import random
import requests
import time

from requests.auth import HTTPBasicAuth

from helper import pprint


def _load_thumbnail(thb_url, api_key, width, password):
    params = {'width': width}
    response = requests.get(thb_url, auth=HTTPBasicAuth(api_key, password), params=params)
    if response.status_code == 200:
        return response.content
    else:
        raise requests.HTTPError(
            f"Cannot load thumbnail for {thb_url}, status code: {response.status_code}, {response.text}")


def _save_thumbnail(tile_id, content, path):
    os.makedirs(path, exist_ok=True)
    with open(f"{path}{tile_id}.png", "wb") as file:
        file.write(content)


def store_thumbnails(items, api_key, width, path, verbose, password='', tries=2):

    for item in items:
        thb_url = item['thumbnail']
        for try_ in range(tries):
            try:
                content = _load_thumbnail(thb_url, api_key, width, password)
                _save_thumbnail(item['id'], content, path)
                break
            except Exception as err:
                pause = random.uniform(0.5, 1.9)
                pprint(f"Cannot get content for {item['id']}, {str(err)}, next try in {pause} sec", verbose)
                time.sleep(pause)
