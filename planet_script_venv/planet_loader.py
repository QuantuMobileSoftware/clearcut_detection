#!/usr/bin/python

import json
import time as t

import easyargs
from planet import api

from downloader import download, quota
from helper import extract_results, overlap, get_best_items, print_item, pprint
from thumbnail import store_thumbnails


@easyargs
def main(cred, input, path="thumbnails/",
         width=512, cloud_percent=30.0, overlap_percent=5.0,
         load=False, item_type="PSOrthoTile", asset_type="analytic", directory="assets/",
         sleep=0.1, tries=3, cores=2, verbose=False):
    """
    Script for searching and downloading best quality tiles using Planet API

    :param cred: path to .json file with Planet API key, example: {"api_key": "value"}
    :param input: path to .json file with request data, default: data.json
    :param path: path to folder where thumbnails wii be stored default: thumbnails/
    :param width: a thumbnail size, default: 512x512. Can be scaled up to 2048 if you have access to the visual asset.
    :param cloud_percent: max cloud coverage, default: 30.0
    :param overlap_percent: min overlap coverage, default 5.0
    :param load: if load real tiles, default False
    :param item_type: Planet item type, default PSOrthoTile
    :param asset_type: Planet asset type, default analytic
    :param directory: directory to save downloaded tiles, default assets/
    :param sleep: wait between pooling Planet API about activated assets in min, default 5
    :param tries: number of pooling tries, default 3
    :param cores: number of cores for pooling assets activation, default 2
    :param verbose: verbose mode
    """

    start = t.time()
    pprint(f"Start script execution", verbose)

    with open(cred) as cred:
        api_key = json.load(cred)["api_key"]

    with open(input) as data:
        request_data = json.load(data)

    client = api.ClientV1(api_key)

    items = list()
    for index, data in enumerate(request_data):

        request = api.filters.build_search_request(**data)
        results = client.quick_search(request)

        result_list = extract_results(results)

        if not result_list:
            print(f"Items not found for index {index} in input file")
        else:
            pprint(f"\nFound {len(result_list)} items", verbose)

            result_list = overlap(data, result_list, overlap_percent, verbose)
            best_items = get_best_items(result_list, cloud_percent)

            if best_items:

                items.extend(best_items)

                pprint("Loading and saving thumbnails of best quality items", verbose)
                store_thumbnails(best_items, api_key, width, path, verbose)

                print(f"\nBEST QUALITY ITEMS for index {index} in input file:\n")

                best_items = sorted(best_items, key=lambda value: (value['overlap_percent'], value['cloud_percent']),
                                    reverse=True)

                for item in best_items:
                    print_item(item)
            else:
                print("Quality items not found. Try to change overlap or cloud percent\n")

    if load and items:
        quota(api_key, verbose)
        download(api, client, [items[0]], item_type, asset_type,
                 directory, sleep, tries, cores, verbose)

    pprint(f"\nFinished execution at {t.strftime('%H:%M:%S', t.gmtime(t.time() - start))}", verbose)


if __name__ == '__main__':
    main()
