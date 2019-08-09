import json

from django.core.exceptions import ObjectDoesNotExist
from django.core.serializers import serialize
from django.http import JsonResponse, HttpResponse
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import JSONParser

from .models import Clearcut
from .serializers import ClearcutChartSerializer
import random


@api_view(['GET'])
def clearcuts_info(request, start_date, end_date):
    """
    Returns geojson with clearcuts, where each feature in geojson has pk, image_date and color property based on polygon
    change during start_date - end_date period.
    """
    if Clearcut.objects.all().count() == 0:
        return HttpResponse(status=404)
    latest_clearcut = Clearcut.objects.filter(image_date__range=[start_date, end_date]).latest('image_date')
    clearcuts = Clearcut.objects.filter(image_date=latest_clearcut.image_date)
    data = serialize('geojson', clearcuts,
                     geometry_field='mpoly',
                     fields=('image_date', 'pk'))

    geojson_dict = json.loads(data)

    for feature in geojson_dict["features"]:
        feature["properties"]["color"] = random.randint(0, 2)
    response = JsonResponse(json.dumps(geojson_dict), safe=False)
    return response


@api_view(['GET'])
def clearcut_area(request, id):
    """
    Returns an area of a chosen clearcut.
    """
    try:
        clearcut = Clearcut.objects.all().get(pk=id)
        result = {
            "area": clearcut.area
        }
        response = JsonResponse(json.dumps(result), safe=False)
        return response
    except ObjectDoesNotExist:
        return HttpResponse(status=404)


@api_view(['GET'])
@parser_classes([JSONParser])
def clearcut_area_chart(request, id, start_date, end_date):
    """
    Returns an array of dictionaries with clearcut date and area during chosen period.
    Values are based on clearcuts from the same zone as chosen polygon.
    """
    try:
        clearcut = Clearcut.objects.all().get(pk=id)
        zone_clearcuts = Clearcut.objects.filter(zone=clearcut.zone).filter(
            image_date__range=[start_date, end_date]).order_by('image_date')
        serializer = ClearcutChartSerializer(zone_clearcuts, many=True)
        return JsonResponse(serializer.data, safe=False)
    except ObjectDoesNotExist:
        return HttpResponse(status=404)
