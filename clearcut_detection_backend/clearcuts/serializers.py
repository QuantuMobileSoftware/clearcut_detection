from .models import Clearcut
from rest_framework import serializers


class ClearcutChartSerializer(serializers.ModelSerializer):
    zone_area = serializers.FloatField()

    class Meta:
        model = Clearcut
        fields = ['image_date', 'zone_area']
