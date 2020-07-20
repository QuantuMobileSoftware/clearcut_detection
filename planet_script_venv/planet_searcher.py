#!/usr/bin/python

import time as t
import easyargs
import geopandas as gp
from planet import api
from search.helper import *
from search.thumbnail import store_thumbnails

from dateutil.relativedelta import relativedelta
from datetime import datetime


@easyargs
def main(credentials,
         input,
         start=(datetime.utcnow().date() - relativedelta(days=7)).strftime('%Y-%m-%d'),
         end=datetime.utcnow().date().strftime('%Y-%m-%d'),
         thumbnails_dir="thumbnails", output="load_assets.json",
         width=512, cloud_percent=100.0, overlap_percent=5.0, verbose=True
         ):
    """
    Script for searching best quality PSOrthoTile tiles using Planet API. !Not implemented for other items_types!

    :param credentials: path to .json file with Planet API key, example: {"api_key": "value"}
    :param input: path to .geojson file with request data
    :param start: start date to search, default: today - 7 days, example: "2020-01-01"
    :param end: end date to search, default: today, example: "2020-01-02"
    :param thumbnails_dir: path to folder where thumbnails wii be stored default: thumbnails
    :param output: path to .json file to store best items. Can be passed in planet_loader.py, default: load_assets.json
    :param width: a thumbnail size, default: 512x512. Can be scaled up to 2048 if you have access to the visual asset.
    :param cloud_percent: max cloud coverage, default: 30.0
    :param overlap_percent: min overlap coverage, default 5.0
    :param verbose: flag, verbose mode
    """

    start_time = t.time()

    """credentials = "credentials.json"
    input = "input/b.geojson"
    start = "2020-07-01"
    end = "2020-07-14" 
    """

    pprint(f"Start script execution\n"
           f"start date: {start}\n"
           f"end_date: {end}\n", verbose)

    with open(credentials) as credentials:
        api_key = json.load(credentials)["api_key"]

    request_df = gp.read_file(input)
    geometry = get_agg_polygon(request_df)

    client = api.ClientV1(api_key)

    request = create_request(geometry, start, end)
    results = client.quick_search(request)
    result_list = extract_results(results)

    if not result_list:
        print(f"Items not found for input Polygon")
    else:
        pprint(f"\nFound {len(result_list)} items", verbose)

        result_list = overlap(geometry, result_list, overlap_percent, verbose)
        best_items = get_best_items(result_list, cloud_percent)

        if best_items:

            pprint("Loading and saving thumbnails of best quality items", verbose)
            store_thumbnails(best_items, api_key, width, thumbnails_dir, verbose)

            print(f"\nBEST QUALITY ITEMS in input file:\n")
            best_items = sorted(best_items, key=lambda value: (value['overlap_percent'], value['cloud_percent']),
                                reverse=True)

            for item in best_items:
                print_item(item)

            write_items(best_items, output)
        else:
            print("Quality items not found. Try to change input args\n")

    pprint(f"\nFinished execution at {t.strftime('%H:%M:%S', t.gmtime(t.time() - start_time))}", verbose)


if __name__ == '__main__':
    main()
