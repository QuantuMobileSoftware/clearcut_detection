import json
import re

from django.db.models import Max, Subquery, Sum, Min, F, OuterRef
from django.core.exceptions import ObjectDoesNotExist
from django.http import JsonResponse, HttpResponse
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import JSONParser

from clearcuts.models import Clearcut, Zone
from clearcuts.serializers import ClearcutChartSerializer


@api_view(['GET'])
def clearcuts_info(request, start_date, end_date):
    """
    Returns geojson with clearcuts, where each feature in geojson has pk and color property based on polygon
    change during start_date - end_date period.
    """
    pattern = "[0-9]{4}-[0-9]{2}-[0-9]{2}"
    if not re.match(pattern, start_date) or not re.match(pattern, end_date):
        return HttpResponse(status=400)
    CHANGED = 0
    UNCHANGED = 1
    NO_DATA = 2

    if Clearcut.objects.all().count() == 0:
        return JsonResponse({})

    date_filtered_clearcuts = Clearcut.objects.filter(image_date__range=[start_date, end_date])
    if date_filtered_clearcuts.count() == 0:
        return JsonResponse({})

    zone_max_min_date_clearcuts = \
        date_filtered_clearcuts \
            .values('zone__id') \
            .annotate(min_date=Min('image_date')) \
            .annotate(max_date=Max('image_date')) \
            .filter(max_date__gt=F('min_date'))

    zone_max_min_date_clearcuts_unchanged = \
        date_filtered_clearcuts \
            .values('zone__id') \
            .annotate(min_date=Min('image_date')) \
            .annotate(max_date=Max('image_date')) \
            .filter(max_date=F('min_date'))

    ordered_clearcuts = Clearcut.objects.filter(zone=OuterRef('pk')).order_by('-image_date', '-area')
    newest_zone_clearcuts = Zone.objects \
        .annotate(newest_clearcut_date=Subquery(ordered_clearcuts.values('image_date')[:1])) \
        .annotate(newest_clearcut_poly=Subquery(ordered_clearcuts.values('mpoly')[:1])) \
        .annotate(newest_clearcut_pk=Subquery(ordered_clearcuts.values('pk')[:1])) \
        .values('pk', 'newest_clearcut_date', 'newest_clearcut_poly', 'newest_clearcut_pk')

    filtered_by_dates_zone_ids = date_filtered_clearcuts \
        .filter(zone__id__in=zone_max_min_date_clearcuts.values_list('zone__id', flat=True)) \
        .values('area', 'zone__id', 'image_date')

    zone_date_area = {}
    unchanged_zones = set()
    changed_zones = set()
    for zone_info in zone_max_min_date_clearcuts_unchanged:
        unchanged_zones.add(zone_info['zone__id'])
    for zone_info in zone_max_min_date_clearcuts:
        zone_date_area[zone_info['zone__id']] = {
            'min_date': zone_info['min_date'],
            'max_date': zone_info['max_date'],
            'min_date_area': 0,
            'max_date_area': 0
        }
    for clearcut in filtered_by_dates_zone_ids:
        if clearcut['image_date'] == zone_date_area[clearcut['zone__id']]['min_date']:
            zone_date_area[clearcut['zone__id']]['min_date_area'] += clearcut['area']
        if clearcut['image_date'] == zone_date_area[clearcut['zone__id']]['max_date']:
            zone_date_area[clearcut['zone__id']]['max_date_area'] += clearcut['area']
    for zone_id in zone_date_area:
        if zone_date_area[zone_id]['min_date_area'] < zone_date_area[zone_id]['max_date_area']:
            changed_zones.add(zone_id)
        else:
            unchanged_zones.add(zone_id)

    geojson_dict = {
        'type': 'FeatureCollection',
        'crs': {
            'type': 'name',
            'properties': {'name': 'EPSG:4326'}
        },
        'features': []
    }
    for newest_clearcut in newest_zone_clearcuts:
        if newest_clearcut['newest_clearcut_poly'].area == 0.0:
            continue
        zone_pk = newest_clearcut['pk']
        poly_pk = newest_clearcut['newest_clearcut_pk']
        geojson = newest_clearcut['newest_clearcut_poly'].geojson
        feature = {
            'type': 'Feature',
            "geometry": json.loads(geojson),
            'properties': {
                'pk': poly_pk
            }
        }
        if zone_pk in changed_zones:
            feature['properties']['color'] = CHANGED
        elif zone_pk in unchanged_zones:
            feature['properties']['color'] = UNCHANGED
        else:
            feature['properties']['color'] = NO_DATA
        geojson_dict['features'].append(feature)
    response = JsonResponse(geojson_dict, safe=False)
    return response


@api_view(['GET'])
@parser_classes([JSONParser])
def clearcut_area_chart(request, id, start_date, end_date):
    """
    Returns an array of dictionaries with clearcut date and area during chosen period.
    Values are based on clearcuts from the same zone as chosen polygon.
    """
    pattern = "[0-9]{4}-[0-9]{2}-[0-9]{2}"
    if not re.match(pattern, start_date) or not re.match(pattern, end_date):
        return HttpResponse(status=400)
    try:
        clearcut = Clearcut.objects.all().get(pk=id)
        zone_clearcuts = Clearcut.objects.filter(zone=clearcut.zone).filter(
            image_date__range=[start_date, end_date]).order_by('image_date') \
            .values('image_date', 'zone_id').annotate(zone_area=Sum('area'))
        serializer = ClearcutChartSerializer(zone_clearcuts, many=True)
        return JsonResponse(serializer.data, safe=False)
    except ObjectDoesNotExist:
        return HttpResponse(status=404)
