import sys
import argparse
import json

from planet import api
from operator import itemgetter, attrgetter
 
 
def createParser():
    parser = argparse.ArgumentParser()
    parser.add_argument ('file', nargs='?')
    return parser

def result_to_list(result_list):
  return [
    (item['id'], 
    item['properties']['visible_percent'], 
    item['properties']['cloud_percent'])
    for item in result_list.items_iter(100) if 'visible_percent' in item['properties']]

def get_best_item(items):
  # Sort by best visible_percent
  result_list = sorted(items, key=itemgetter(1), reverse=True)  
  # Get list with only best visible_percent
  result_list = list(filter(lambda item: item[1] >= result_list[0][1], result_list))
  # Sort by lowest cloud_percent
  result_list = sorted(result_list, key=itemgetter(2))
  return result_list[0]

if __name__ == "__main__":
  parser = createParser()
  namespace = parser.parse_args (sys.argv[1:])
  with open(namespace.file) as data:
    request_data = json.load(data)

  client = api.ClientV1()

  for data in request_data:
    request = api.filters.build_search_request(**data)
    results = client.quick_search(request)
    result_list = result_to_list(results)
    if result_list == []:
      sys.stdout.write('%s\n' % 'Not found items.')
    else: 
      best_quality_item = get_best_item(result_list)
      sys.stdout.write('%s\n' % best_quality_item[0])
