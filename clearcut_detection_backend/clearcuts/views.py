from django.core.serializers import serialize
from django.http import JsonResponse
from rest_framework.decorators import api_view
from rest_framework import viewsets
from django.contrib.gis.db.models.functions import Area, AsGeoJSON, Transform
from .models import Clearcut
from django.db.models import Max, F
from .serializers import ClearcutSerializer
import json


class ClearcutViewSet(viewsets.ModelViewSet):
    queryset = Clearcut.objects.all()
    serializer_class = ClearcutSerializer


@api_view()
def clearcuts_info(request):
    if request.method == 'GET':
        latest_clearcut = Clearcut.objects.latest('image_date')
        clearcuts = Clearcut.objects.filter(image_date=latest_clearcut.image_date)
        data = serialize('geojson', clearcuts,
                         geometry_field='mpoly',
                         fields=('image_date', 'pk'))
        response = JsonResponse(data, safe=False)
        response["Access-Control-Allow-Origin"] = "*"
        return response


@api_view()
def clearcut_area(request, pk):
    if request.method == 'GET':
        clearcut = Clearcut.objects.annotate(transformed_poly=Transform('mpoly', 3857)).get(pk=pk)
        poly = clearcut.transformed_poly
        area = poly.area
        result = {
            "area": area
        }
        response = JsonResponse(json.dumps(result), safe=False)
        response["Access-Control-Allow-Origin"] = "*"
        return response
