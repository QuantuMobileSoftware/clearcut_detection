#!/usr/bin/python

import time
import easyargs

from load.helper import quota, read_items, check_loaded, get_auth, save_not_loaded
from load.load import download
from load.order import poll_order_mult, post_orders
from search.helper import pprint


@easyargs
def main(credentials,
         input,
         item_type="PSOrthoTile",
         asset_type="analytic",
         directory="assets",
         output="not_loaded.json",
         tries=-1, cores=2, verbose=True, delay=180):
    """
    Script for downloading best quality tiles using Planet API. Be careful, quota is used, when the ORDER is created

    :param credentials: path to .json file with Planet API key, example: {"api_key": "value"}
    :param input: path to .json file with request tiles metadata, example: {"id": {}, "id": {}}
    :param item_type: Planet item type, default PSOrthoTile
    :param asset_type: Planet asset type, default analytic
    :param directory: directory to save downloaded tiles, default assets
    :param output: path to .json file to save not loaded assets metadata
    :param tries: number of pooling tries, default -1, infinite (waiting while an asset will be ready to download)
    :param delay: seconds between polls, default 180 sec
    :param cores: number of cores for pooling assets activation, default 2
    :param verbose: verbose mode, default=True
    """

    start = time.time()
    pprint(f"Start script execution\n"
           f"Loading {item_type} assets and {asset_type} type", verbose)

    auth = get_auth(credentials)
    quota(auth, verbose)

    items = read_items(input)
    items = check_loaded(items, directory, verbose)

    if not items:
        print("All items exists. Nothing to load. Check directories")
        return

    items = post_orders(auth, items, item_type, asset_type, verbose)
    items = poll_order_mult(items, auth, cores, tries, delay, verbose)

    not_loaded = download(items, directory, verbose)

    if not_loaded:
        print(f"Not loaded items:\n{sorted(not_loaded.keys())}")
        save_not_loaded(not_loaded, output)

    pprint(f"\nFinished execution at {time.strftime('%H:%M:%S', time.gmtime(time.time() - start))}", verbose)


if __name__ == '__main__':
    main()
