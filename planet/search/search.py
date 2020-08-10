from shapely.geometry import Polygon, mapping
from planet.api import filters
from datetime import datetime

from search.helper import pprint


def extract_results(results, limit=100):
    items = [
        {'id': item['id'],
         'visible_percent': item['properties']['visible_percent'],
         'cloud_percent': item['properties']['cloud_percent'],
         'thumbnail': item['_links']['thumbnail'],
         'coordinates': item['geometry']['coordinates'][0],
         }
        for item in results.items_iter(limit) if 'visible_percent' in item['properties']]

    # for item in results.items_iter(limit):
    # print(item)

    return items


def overlap(geometry, result_list, overlap_percent, verbose):
    target_polygon = Polygon(geometry["coordinates"][0])

    items = list()
    for item in result_list:

        item_coordinates = item["coordinates"]

        try:
            if len(item_coordinates) == 1:
                pprint(f"For {item['id']} Planet API returned nested coordinates {item_coordinates}. Extracting them.")
                item_coordinates = item_coordinates[0]

            item_polygon = Polygon(item_coordinates)
            percent = round((target_polygon.intersection(item_polygon).area / target_polygon.area) * 100, 1)

            if percent < overlap_percent:
                pprint(f"Item {item['id']} has small intersection percent {percent}", verbose)
            else:
                item['overlap_percent'] = percent
                items.append(item)
        except Exception as e:
            pprint(f"Cannot crete Polygon for {item['id']}: {str(e)}", verbose)

    return items


def get_best_items(items, cloud_percent):
    # Sort by max visible_percent
    result_list = sorted(items, key=lambda item: item['visible_percent'], reverse=True)
    # Get only best visible_percent
    result_list = list(filter(lambda item: item['visible_percent'] >= result_list[0]['visible_percent'], result_list))

    # Get items with min cloud_percent
    result_list = list(filter(lambda item: item['cloud_percent'] <= cloud_percent, result_list))

    result_list = sorted(result_list, key=lambda item: (item['overlap_percent'], item['cloud_percent']),
                         reverse=True)

    return result_list


def create_request(geometry, start, end, item_types=["PSOrthoTile"]):
    geometry_filter = {
        "type": "GeometryFilter",
        "field_name": "geometry",
        "config": geometry,
    }

    start_date = datetime.strptime(start, '%Y-%m-%d')
    end_date = datetime.strptime(end, '%Y-%m-%d').replace(hour=23, minute=59, second=59)
    date_filter = filters.date_range("acquired", gte=start_date, lte=end_date)

    filter = filters.and_filter(date_filter, geometry_filter)
    request = filters.build_search_request(filter, item_types)

    return request


def get_agg_polygon(request_df):
    bounds = request_df.bounds
    minx, miny = bounds[["minx", "miny"]].min()
    maxx, maxy = bounds[["maxx", "maxy"]].max()
    polygon = Polygon.from_bounds(minx, miny, maxx, maxy)
    return mapping(polygon)
