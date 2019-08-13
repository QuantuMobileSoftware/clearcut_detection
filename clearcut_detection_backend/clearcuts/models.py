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
