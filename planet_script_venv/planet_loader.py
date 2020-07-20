#!/usr/bin/python

import time as t
import easyargs

from load.helper import quota, read_items, check_loaded, get_auth, save_not_loaded
from load.load import download
from load.order import poll_order_mult, post_orders
from search.helper import pprint


@easyargs
def main(cred,
         input,
         item_type="PSOrthoTile",
         asset_type="analytic",
         directory="assets",
         output="not_loaded.json",
         tries=-1, cores=2, verbose=True, delay=180):
    """
    Script for downloading best quality tiles using Planet API. Be careful, quota is used, when the ORDER is created

    :param cred: path to .json file with Planet API key, example: {"api_key": "value"}
    :param input: path to .json file with request tiles, example: {"id": {}, "id": {}}
    :param item_type: Planet item type, default PSOrthoTile
    :param asset_type: Planet asset type, default analytic
    :param directory: directory to save downloaded tiles, default assets
    :param output: path to .json file to save not loaded assets metadata
    :param tries: number of pooling tries, default -1, infinite (while asset will be ready to download)
    :param delay: seconds between polls, default 180 sec
    :param cores: number of cores for pooling assets activation, default 2
    :param verbose: verbose mode
    """

    start = t.time()
    pprint(f"Start script execution\n"
           f"Loading {item_type} assets and {asset_type} type", verbose)

    """cred = "credentials.json"
       input = 'input/load_kharkiv.json'
       item_type = "Sentinel2L1C"
    """

    auth = get_auth(cred)
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

    pprint(f"\nFinished execution at {t.strftime('%H:%M:%S', t.gmtime(t.time() - start))}", verbose)


if __name__ == '__main__':
    main()

