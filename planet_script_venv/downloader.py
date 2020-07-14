import os
import time
import requests
from requests.auth import HTTPBasicAuth
from concurrent.futures import ThreadPoolExecutor, as_completed
from helper import pprint


def get_items_assets(client, items, item_type, asset_type, verbose):
    items_assets = dict()
    for item in items:
        item_assets = client.get_assets_by_id(item_type, item["id"]).get()
        item_assets = item_assets.get(asset_type)
        if not item_assets:
            pprint(f"Cannot get assets for {item['id']} from Planet API. Check api_key and permissions to download",
                   verbose)
        else:
            items_assets[item["id"]] = dict(assets=item_assets)

    return items_assets


def activate_mult(items_assets, client, sleep, tries, cores, verbose):
    pprint("Activating assets", verbose)

    with ThreadPoolExecutor(max_workers=cores) as executor:
        # Start the pooling Planet API for next loading
        future_to_activate = {
            executor.submit(_activate, client, items_assets[item_id]["assets"], item_id, sleep, tries, verbose):
                item_id for item_id in items_assets.keys()}

        for future in as_completed(future_to_activate):
            item_id = future_to_activate[future]
            try:
                is_activated = future.result()
                items_assets[item_id]["activated"] = is_activated
            except Exception as exc:
                pprint(f"Generated exception for {item_id}: {str(exc)}", verbose)
    return items_assets


def _activate(client, item_assets, item_id, sleep, tries, verbose):
    for try_ in range(0, tries):
        activation = client.activate(item_assets)
        status_code = activation.response.status_code

        # If asset is already active, we are done
        if status_code == 204:
            pprint(f"Asset for {item_id} ready to download, status code: {status_code}", verbose)
            return True
        elif status_code == 401:
            pprint(f"You don't have permission for {item_id} to download, status code: {status_code}", verbose)
            return False
        else:
            # Still activating. Wait and check again.
            pprint(f"Waiting for {item_id} asset activation, status code: {status_code}...", verbose)
            if try_ != tries - 1:
                time.sleep(sleep * 60)

    return False


def load_assets(api, client, assets, directory, verbose):
    os.makedirs(directory, exist_ok=True)

    for id, asset in assets.items():
        if asset.get("activated"):
            try:
                request_asset = assets[id]["assets"]
                pprint(f"Downloading {id} asset", verbose)
                callback = api.write_to_file(directory, callback=downloaded)
                body = client.download(request_asset, callback=callback)

            except Exception as e:
                pprint(f"cannot download {id} asset: {str(e)}", verbose)
        break


def download(api, client, items, item_type, asset_type, directory, sleep, tries, cores, verbose):
    items_assets = get_items_assets(client, items, item_type, asset_type, verbose)
    if not items_assets:
        print(f"No assets to download")
        return

    activated_assets = activate_mult(items_assets, client, sleep, tries, cores, verbose)

    pprint(f"Activated {len(activated_assets)} assets", verbose)

    load_assets(api, client, activated_assets, directory, verbose)


# TODO: deal with callback for loading
def downloaded(callback):
    # print(callback)
    pass


def quota(api_key, verbose):
    """Print allocation and remaining quota"""

    response = requests.get(
        "https://api.planet.com/auth/v1/experimental/public/my/subscriptions",
        auth=HTTPBasicAuth(api_key, ''),
    )

    if response.status_code == 200:
        response = response.json()[0]

        quota_sqkm = float(response["quota_sqkm"])
        quota_used = round(float(response["quota_used"]), 2)

        pprint(f"\nQuota status:\nOrganization name: {response['organization']['name']}\n"
               f"Quota enabled: {response['quota_enabled']}\n"
               f"Total quota in SqKm: {quota_sqkm}\n"
               f"Total quota used: {quota_used}", verbose)

        left_quota = round(quota_sqkm - quota_used, 2)
        try:
            left_percent = round((left_quota * 100.0) / quota_sqkm, 2)
            if left_percent < 5.0:
                print(f"Warning: left {left_percent}% of total quota to load")
        except ZeroDivisionError as ex:
            pprint(f"Cannot calculate left quota percent: {str(ex)}", verbose)
            pprint(f"Remaining Quota: {left_quota} SqKm", verbose)
        else:
            pprint(f"Remaining Quota: {left_quota} SqKm or {left_percent}%", verbose)

    else:
        pprint(f"Cannot get quota status: {response.status_code}", verbose)
