import json
import os

import requests
from requests.auth import HTTPBasicAuth


def pprint(text, verbose=True):
    if verbose:
        print(text)


def get_auth(path):
    with open(path) as cred:
        api_key = json.load(cred)["api_key"]

    return HTTPBasicAuth(api_key, '')


def quota(auth, verbose):
    """Print allocation and remaining quota"""

    response = requests.get(
        "https://api.planet.com/auth/v1/experimental/public/my/subscriptions",
        auth=auth,
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
        except ZeroDivisionError as ex:
            pprint(f"Cannot calculate left quota percent: {str(ex)}"
                   f"Remaining Quota: {left_quota} SqKm", verbose)
        else:
            if left_percent < 5.0:
                print(f"Warning: left {left_percent}% of total quota to load")
            else:
                pprint(f"Remaining Quota: {left_quota} SqKm or {left_percent}%", verbose)

    else:
        pprint(f"Cannot get quota status: {response.status_code}", verbose)


def read_items(input):
    with open(input) as f:
        return json.load(f)


def check_loaded(items, directory, verbose):
    if not os.path.exists(directory):
        pprint(f"Path to assets not exist. All items will be loaded", verbose)
        return items

    for item in list(items):
        path = f"{directory}/{item}"
        pprint(f"Checking cache in {path}", verbose)
        if os.path.exists(path):
            pprint(f"Path to {path} exists. Asset will not be loaded", verbose)
            items.pop(item)

    pprint(f"{len(items)} items will be loaded", verbose)
    return items


def save_asset(item_id, name, content, directory, verbose):
    if "." in name:
        name = name.split("/")[-1]
    else:
        name = f"{item_id}.zip"

    path = f"{directory}/{item_id}/"
    os.makedirs(path, exist_ok=True)

    save_path = f"{path}{name}"

    with open(save_path, "wb") as file:
        file.write(content)

    pprint(f"Item {name} saved to {save_path}", verbose)


def save_not_loaded(items, output):
    with open(output, 'w') as f:
        json.dump(items, f, indent=1)
