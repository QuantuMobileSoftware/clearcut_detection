from django.db import models
from clearcuts.models import Tile
from django.utils import timezone


class SourceJp2Images(models.Model):
    tile = models.ForeignKey(Tile, on_delete=models.CASCADE, related_name='jp2_images')
    image_date = models.DateField(null=False)
    tile_uri = models.URLField(max_length=200, unique=True)
    cloud_coverage = models.FloatField(default=0)
    nodata_pixel = models.FloatField(default=0)
    source_tci_location = models.URLField(max_length=200, blank=True, null=True)
    source_b04_location = models.URLField(max_length=200, blank=True, null=True)
    source_b08_location = models.URLField(max_length=200, blank=True, null=True)
    source_b8a_location = models.URLField(max_length=200, blank=True, null=True)
    source_b11_location = models.URLField(max_length=200, blank=True, null=True)
    source_b12_location = models.URLField(max_length=200, blank=True, null=True)
    source_clouds_location = models.URLField(max_length=200, blank=True, null=True)

    is_downloaded = models.SmallIntegerField(default=0)
    check_date = models.DateTimeField(default=timezone.now)

    class Meta:
        unique_together = [['tile', 'image_date']]
