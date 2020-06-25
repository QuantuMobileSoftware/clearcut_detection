import sys
import argparse
import json
from planet import api
from operator import itemgetter, attrgetter
from requests.auth import HTTPBasicAuth
import random
import time
import os
import requests
from shapely.geometry import Polygon

from helper import result_to_dict, overlap, get_best_items, print_item
from thumbnail import store_thumbnails
import easyargs


@easyargs
def main(api_key, input_file="data.json", verbose=False):
    path = "data.json"

    with open(path) as data:
        request_data = json.load(data)

    client = api.ClientV1(api_key)

    for data in request_data:
        request = api.filters.build_search_request(**data)
        results = client.quick_search(request)

        result_list = result_to_dict(results)

        if not result_list:
            print('Items not found.')
        else:
            print(f"Found {len(result_list)} items")

            result_list = overlap(data, result_list)

            best_quality_items = get_best_items(result_list)

            if best_quality_items:
                print("Loading and saving thumbnails of best items")
                store_thumbnails(best_quality_items, api_key)

                print("\nBEST QUALITY ITEMS:\n")
                for item in best_quality_items:
                    print_item(item)
                print("\n")
            else:
                print("No items found satisfying the condition")


if __name__ == '__main__':
    main()
