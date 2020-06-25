from django.contrib.gis.db import models
from django.utils import timezone

class Zone(models.Model):
    def __str__(self):
        return f"Zone {self.id}"


class Clearcut(models.Model):
    image_date_0 = models.DateField(default=timezone.now())
    image_date_1 = models.DateField(default=timezone.now())
    area = models.FloatField()
    forest = models.PositiveIntegerField(default=1)
    centroid = models.PointField()
    zone = models.ForeignKey(Zone, on_delete=models.CASCADE)
    mpoly = models.PolygonField()

    def __str__(self):
        return f"Clearcut {self.id}"


class TileInformation(models.Model):
    tile_name   = models.CharField(max_length=7, blank=False, null=False)
    tile_index  = models.CharField(max_length=5, blank=False, null=False)
    tile_date   = models.DateField(default=timezone.now())

    tile_location = models.URLField(max_length=200, blank=True, null=True)
    source_tci_location = models.URLField(max_length=200, blank=True, null=True)
    source_b04_location = models.URLField(max_length=200, blank=True, null=True)
    source_b08_location = models.URLField(max_length=200, blank=True, null=True)
    source_b8a_location = models.URLField(max_length=200, blank=True, null=True)
    source_b11_location = models.URLField(max_length=200, blank=True, null=True)
    source_b12_location = models.URLField(max_length=200, blank=True, null=True)
    source_clouds_location = models.URLField(max_length=200, blank=True, null=True)
    model_tiff_location = models.URLField(max_length=200, blank=True, null=True)

    tile_metadata_hash = models.CharField(max_length=32, default=0, blank=True, null=True)
    cloud_coverage = models.FloatField(default=0, blank=False, null=False)
    mapbox_tile_id = models.CharField(max_length=32, blank=True, null=True)
    mapbox_tile_name = models.CharField(max_length=32, blank=True, null=True)
    mapbox_tile_layer = models.CharField(max_length=32, blank=True, null=True)
    coordinates = models.PolygonField(blank=True, null=True)

    objects = models.Manager()