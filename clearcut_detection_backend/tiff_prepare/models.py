from django.db import models
from clearcuts.models import Tile
from django.utils import timezone


class Prepared(models.Model):
    tile = models.ForeignKey(Tile, on_delete=models.CASCADE, related_name='tiff_prepare')
    image_date = models.DateField(null=False)
    model_tiff_location = models.URLField(max_length=200, blank=True, null=True)
    cloud_tiff_location = models.URLField(max_length=200, blank=True, null=True)
    success = models.SmallIntegerField(default=0)
    is_new = models.SmallIntegerField(default=1)
    prepare_date = models.DateTimeField(default=timezone.now)

    class Meta:
        unique_together = [['tile', 'image_date']]
