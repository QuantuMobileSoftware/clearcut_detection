from django.contrib.gis.db import models


class Zone(models.Model):
    def __str__(self):
        return f"Zone {self.id}"


class Clearcut(models.Model):
    image_date = models.DateField()
    area = models.FloatField()
    centroid = models.PointField()
    zone = models.ForeignKey(Zone, on_delete=models.CASCADE)
    mpoly = models.PolygonField()

    def __str__(self):
        return f"Clearcut {self.id}"


class TileInformation(models.Model):
    tile_name = models.CharField(max_length=5, blank=False, null=False)

    tile_location = models.CharField(max_length=60, blank=True, null=True)
    source_tci_location = models.CharField(max_length=60, blank=True, null=True)
    source_b04_location = models.CharField(max_length=60, blank=True, null=True)
    source_b08_location = models.CharField(max_length=60, blank=True, null=True)
    model_tiff_location = models.CharField(max_length=60, blank=True, null=True)

    tile_metadata_hash = models.CharField(max_length=32, default=0, blank=True, null=True)
    cloud_coverage = models.FloatField(default=0, blank=False, null=False)
    mapbox_tile_id = models.CharField(max_length=32, blank=True, null=True)
    mapbox_tile_name = models.CharField(max_length=32, blank=True, null=True)
    mapbox_tile_layer = models.CharField(max_length=32, blank=True, null=True)
    coordinates = models.PolygonField(blank=True, null=True)

    objects = models.Manager()
