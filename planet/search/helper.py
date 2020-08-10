import json


def print_item(item):
    print(f"id: {item['id']}, "
          f"overlap percent: {item['overlap_percent']}, "
          f"visible percent {item['visible_percent']}, "
          f"cloud percent {item['cloud_percent']}")


def write_items(items, output):
    with open(output, 'w') as f:
        items_dict = dict()
        for item in items:
            items_dict[item["id"]] = {"overlap percent": item['overlap_percent'],
                                      "visible percent": item['visible_percent'],
                                      "cloud percent": item['cloud_percent'],
                                      # "coordinates": item['coordinates'],
                                      }

        json.dump(items_dict, f, indent=1)


def pprint(text, verbose=True):
    if verbose:
        print(text)
