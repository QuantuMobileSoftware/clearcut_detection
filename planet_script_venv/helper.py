from shapely.geometry import Polygon


def result_to_dict(results, limit=100):
    items = [
        {'id': item['id'],
         'visible_percent': item['properties']['visible_percent'],
         'cloud_percent': item['properties']['cloud_percent'],
         'thumbnail': item['_links']['thumbnail'],
         'coordinates': item['geometry']['coordinates'][0],
         }
        for item in results.items_iter(limit) if 'visible_percent' in item['properties']]

    return items


def overlap(target, result_list, min_percent=0.01):
    target_polygon = Polygon(target['filter_like']['config'][0]['config']['coordinates'][0])

    for index, item in enumerate(result_list):
        item_polygon = Polygon(item['coordinates'])
        percent = round((target_polygon.intersection(item_polygon).area / target_polygon.area) * 100, 1)
        if percent < min_percent:
            print(f"Item {item['id']} has small intersection percent {percent}")
        item['overlap_percent'] = percent
        result_list[index] = item

    return result_list


def get_best_items(items, cloud_percent=30):
    # Sort by max visible_percent
    result_list = sorted(items, key=lambda item: item['visible_percent'], reverse=True)
    # Get only best visible_percent
    result_list = list(filter(lambda item: item['visible_percent'] >= result_list[0]['visible_percent'], result_list))

    # Get items with min cloud_percent
    result_list = list(filter(lambda item: item['cloud_percent'] <= cloud_percent, result_list))
    result_list = sorted(result_list, key=lambda item: item['cloud_percent'])

    return result_list


def print_item(item):
    print(f"ID: {item['id']}, "
          f"overlap percent: {item['overlap_percent']}, "
          f"visible percent {item['visible_percent']}, "
          f"cloud percent {item['cloud_percent']}")


def pprint(text, verbose=True):
    if verbose:
        print(text)
