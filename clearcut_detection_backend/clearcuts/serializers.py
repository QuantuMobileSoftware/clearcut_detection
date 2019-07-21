from .models import Clearcut
from rest_framework import serializers


class ClearcutSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Clearcut
        fields = ['forest_type', 'forest_state', 'detected_class', 'image_date', 'mpoly']
