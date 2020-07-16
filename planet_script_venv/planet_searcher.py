#!/usr/bin/python

import time as t
import easyargs
from planet import api
from search.helper import *
from search.thumbnail import store_thumbnails

from dateutil.relativedelta import relativedelta
from datetime import datetime


@easyargs
def main(credentials, input,
         start=(datetime.utcnow().date() - relativedelta(days=14)).strftime('%Y-%m-%d'),
         end=datetime.utcnow().date().strftime('%Y-%m-%d'),
         thumbnails_dir="thumbnails", output="load_assets.json",
         width=512, cloud_percent=100.0, overlap_percent=5.0, verbose=True
         ):
    """
    Script for searching best quality PSOrthoTile tiles using Planet API. !Not implemented for other items_types!

    :param credentials: path to .json file with Planet API key, example: {"api_key": "value"}
    :param input: path to .json file with request data
    :param start: start date to search, default: today - 14 days, example: "2020-01-01"
    :param end: end date to search, default: today, example: "2020-01-02"
    :param thumbnails_dir: path to folder where thumbnails wii be stored default: thumbnails
    :param output: path to .json file where best items will be stored, default: load_assets.json
    :param width: a thumbnail size, default: 512x512. Can be scaled up to 2048 if you have access to the visual asset.
    :param cloud_percent: max cloud coverage, default: 30.0
    :param overlap_percent: min overlap coverage, default 5.0
    :param verbose: flag, verbose mode
    """

    start_time = t.time()
    pprint(f"Start script execution", verbose)

    with open(credentials) as credentials:
        api_key = json.load(credentials)["api_key"]

    with open(input) as data:
        request_data = json.load(data)

    client = api.ClientV1(api_key)

    items = list()
    for index, data in enumerate(request_data):

        geometry = data['features'][0]['geometry']
        request = create_request(geometry, start, end)
        results = client.quick_search(request)
        result_list = extract_results(results)

        if not result_list:
            print(f"Items not found for index {index} in input file")
        else:
            pprint(f"\nFound {len(result_list)} items", verbose)

            result_list = overlap(geometry, result_list, overlap_percent, verbose)
            best_items = get_best_items(result_list, cloud_percent)

            if best_items:

                items.extend(best_items)

                pprint("Loading and saving thumbnails of best quality items", verbose)
                store_thumbnails(best_items, api_key, width, thumbnails_dir, verbose)

                print(f"\nBEST QUALITY ITEMS for index {index} in input file:\n")

                best_items = sorted(best_items, key=lambda value: (value['overlap_percent'], value['cloud_percent']),
                                    reverse=True)

                for item in best_items:
                    print_item(item)
            else:
                print("Quality items not found. Try to change overlap or cloud percent\n")

    if items:
        write_items(items, output)
    else:
        print(f"No Quality items found! Nothing to store in {output}")

    pprint(f"\nFinished execution at {t.strftime('%H:%M:%S', t.gmtime(t.time() - start_time))}", verbose)


if __name__ == '__main__':
    """credentials = "credentials.json"
    input = "Kharkiv.geojson"
    """
    main()
