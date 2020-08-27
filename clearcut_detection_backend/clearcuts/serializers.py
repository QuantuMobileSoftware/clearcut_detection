from .models import Clearcut, RunUpdateTask, Zone
from rest_framework import serializers


class ClearcutChartSerializer(serializers.ModelSerializer):
    zone_area = serializers.FloatField()

    class Meta:
        model = Clearcut
        fields = ['image_date_current', 'zone_area']


class ClearcutSerializer(serializers.ModelSerializer):
    image_date_current = serializers.DateField(read_only=True)
    image_date_previous = serializers.DateField(read_only=True)
    # zone = serializers.PrimaryKeyRelatedField(queryset=Zone.objects.all())

    class Meta:
        model = Clearcut
        fields = ['status', 'mpoly', 'image_date_current', 'image_date_previous']


class RunUpdateTaskSerializer(serializers.ModelSerializer):
    result = serializers.URLField(max_length=200, min_length=None, allow_blank=True, allow_null=True)

    class Meta:
        model = RunUpdateTask
        fields = '__all__'
