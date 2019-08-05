import json

from django.core.exceptions import ObjectDoesNotExist
from django.core.serializers import serialize
from django.http import JsonResponse, HttpResponse
from rest_framework import viewsets
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import JSONParser

from .models import Clearcut
from .serializers import ClearcutSerializer, ClearcutChartSerializer


class ClearcutViewSet(viewsets.ModelViewSet):
    queryset = Clearcut.objects.all()
    serializer_class = ClearcutSerializer


@api_view()
def clearcuts_info(request):
    if request.method == 'GET':
        if Clearcut.objects.all().count() == 0:
            return HttpResponse(status=404)
        latest_clearcut = Clearcut.objects.latest('image_date')
        clearcuts = Clearcut.objects.filter(image_date=latest_clearcut.image_date)
        data = serialize('geojson', clearcuts,
                         geometry_field='mpoly',
                         fields=('image_date', 'pk'))
        response = JsonResponse(data, safe=False)
        return response


@api_view()
def clearcut_area(request, pk):
    if request.method == 'GET':
        try:
            clearcut = Clearcut.objects.all().get(pk=pk)
            result = {
                "area": clearcut.area
            }
            response = JsonResponse(json.dumps(result), safe=False)
            return response
        except ObjectDoesNotExist:
            return HttpResponse(status=404)


@api_view(['GET'])
@parser_classes([JSONParser])
def clearcut_area_chart(request):
    try:
        pk = request.query_params['pk']
        start_date = request.query_params['start_date']
        end_date = request.query_params['end_date']
        clearcut = Clearcut.objects.all().get(pk=pk)
        zone_clearcuts = Clearcut.objects.filter(zone=clearcut.zone).filter(
            image_date__range=[start_date, end_date]).order_by('image_date')
        serializer = ClearcutChartSerializer(zone_clearcuts, many=True)
        return JsonResponse(serializer.data, safe=False)
    except ObjectDoesNotExist:
        return HttpResponse(status=404)
