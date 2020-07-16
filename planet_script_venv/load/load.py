import requests

from load.helper import save_asset
from search.helper import pprint


def download(items, directory, verbose):
    pprint("Downloading assets...", verbose)
    not_loaded = dict()
    for item_id, item in items.items():
        results = item.get("results")
        if not results:
            not_loaded[item_id] = item
        else:
            pprint(f"Loading {item_id}", verbose)
            for result in results:
                download_url = result["location"]
                name = result["name"]
                response = requests.get(download_url)

                if response.status_code in (401, 404, 500):
                    pprint(f"Cannot download item {item_id}, status: {response.status_code}, error: {response.json()}",
                           verbose)
                    not_loaded[item_id] = item
                else:
                    pprint(f"Downloaded {item_id}, status: {response.status_code}", verbose)
                    save_asset(item_id, name, response.content, directory, verbose)

    return not_loaded
