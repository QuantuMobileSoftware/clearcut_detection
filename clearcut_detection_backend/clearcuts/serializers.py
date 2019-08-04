from .models import Clearcut
from rest_framework import serializers


class ClearcutSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Clearcut
        fields = ['image_date', 'mpoly']


class ClearcutChartSerializer(serializers.ModelSerializer):
    class Meta:
        model = Clearcut
        fields = ['image_date', 'area']
