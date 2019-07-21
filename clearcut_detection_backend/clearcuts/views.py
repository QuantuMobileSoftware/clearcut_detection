from django.core.serializers import serialize
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from rest_framework import viewsets

from .models import Clearcut
from .serializers import ClearcutSerializer


class ClearcutViewSet(viewsets.ModelViewSet):
    queryset = Clearcut.objects.all()
    serializer_class = ClearcutSerializer


@csrf_exempt
def get_clearcuts(request):
    if request.method == 'GET':
        data = serialize('geojson', Clearcut.objects.all(),
                         geometry_field='mpoly',
                         fields=('forest_type', 'forest_state', 'detected_class', 'image_date',))
        response = JsonResponse(data, safe=False)
        response["Access-Control-Allow-Origin"] = "*"
        return response
