from .models import Clearcut
from rest_framework import serializers


class ClearcutChartSerializer(serializers.ModelSerializer):
    class Meta:
        model = Clearcut
        fields = ['image_date', 'area']
