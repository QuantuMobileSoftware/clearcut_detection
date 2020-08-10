import requests

from retry.api import retry_call
from load.helper import pprint
from concurrent.futures import ThreadPoolExecutor, as_completed


def _post_order(item_id, auth, verbose, item_type, product_bundle):
    order = {
        "name": item_id,
        "products": [
            {
                "item_ids": [
                    item_id
                ],
                "item_type": item_type,
                "product_bundle": product_bundle
            }
        ],
        "delivery": {
            "single_archive": True,
            "archive_type": "zip",
            "archive_filename": item_id
        },
        "order_type": "full"
    }

    url = "https://api.planet.com/compute/ops/orders/v2"

    response = requests.post(url, auth=auth, json=order)
    text = response.json()

    if response.status_code == 202:
        pprint(f"For {item_id}, state: {text['state']}", verbose)
        return text["id"]
    else:
        raise RuntimeError(f"Cannot post order for {item_id}, status_code: {response.status_code}, {text}")


def post_orders(auth, items, item_type, asset_type, verbose):
    for item_id, item in items.items():
        pprint(f"Post order for item_id: {item_id}", verbose)
        try:
            order_id = _post_order(item_id, auth, verbose, item_type, asset_type)
            item["order_id"] = order_id
        except RuntimeError as er:
            ex = str(er)
            pprint(f"Post order, got error: {ex}", verbose)
            item["error"] = ex

    return items


def _poll_order(order_id, auth, verbose):
    if not order_id:
        return

    url = f"https://api.planet.com/compute/ops/orders/v2/{order_id}"
    response = requests.get(url, auth=auth)

    if response.status_code == 200:
        text = response.json()
        state = text["state"]
        pprint(f"Poll message: {text['last_message']} for order_id {order_id}, state: {state}\n", verbose)

        if state == "success":
            pprint(f"Item {text['name']} ready, state: {state}", verbose)
            return text["_links"]["results"]
        elif state in ("failed", "cancelled"):
            pprint(f"Item {text['name']} state: {state}", verbose)
            return
        else:
            raise RuntimeError(f"Item {text['name']} state: {state}")
    else:
        raise RuntimeError(f"Cannot poll order for order_id: {order_id}, status: {response.status_code},"
                           f"message: {response.text}")


def poll_order_mult(items, auth, cores, tries, delay, verbose):
    pprint("Polling orders...", verbose)

    with ThreadPoolExecutor(max_workers=cores) as executor:
        # Start the pooling Planet API for order status
        future_to_activate = {
            executor.submit(retry_call, _poll_order, exceptions=RuntimeError,
                            fargs=[item.get("order_id"), auth, verbose], delay=delay, tries=tries): item_id
            for item_id, item in items.items()}

        for future in as_completed(future_to_activate):
            item_id = future_to_activate[future]
            try:
                items[item_id]["results"] = future.result()
            except Exception as exc:
                pprint(f"Generated exception for {item_id}: {str(exc)}", verbose)
    return items
