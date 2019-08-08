import json

from django.core.exceptions import ObjectDoesNotExist
from django.core.serializers import serialize
from django.http import JsonResponse, HttpResponse
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import JSONParser

from .models import Clearcut
from .serializers import ClearcutChartSerializer


@api_view(['GET'])
def clearcuts_info(request):
    if Clearcut.objects.all().count() == 0:
        return HttpResponse(status=404)
    latest_clearcut = Clearcut.objects.latest('image_date')
    clearcuts = Clearcut.objects.filter(image_date=latest_clearcut.image_date)
    data = serialize('geojson', clearcuts,
                     geometry_field='mpoly',
                     fields=('image_date', 'pk'))
    response = JsonResponse(data, safe=False)
    return response


@api_view(['GET'])
def clearcut_area(request, id):
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
    try:
        clearcut = Clearcut.objects.all().get(pk=id)
        zone_clearcuts = Clearcut.objects.filter(zone=clearcut.zone).filter(
            image_date__range=[start_date, end_date]).order_by('image_date')
        serializer = ClearcutChartSerializer(zone_clearcuts, many=True)
        return JsonResponse(serializer.data, safe=False)
    except ObjectDoesNotExist:
        return HttpResponse(status=404)
