from django.contrib.gis.db import models


class Clearcut(models.Model):
    forest_type = models.CharField(max_length=20)
    forest_state = models.CharField(max_length=20)
    detected_class = models.CharField(max_length=20)
    image_date = models.DateField()

    mpoly = models.PolygonField()

    def __str__(self):
        return str(self.id)
