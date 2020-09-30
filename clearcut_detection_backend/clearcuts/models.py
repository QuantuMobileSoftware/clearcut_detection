from django.contrib.gis.db import models
from django.utils import timezone


class Tile(models.Model):
    tile_index = models.CharField(unique=True, max_length=7, blank=False, null=False)
    is_tracked = models.SmallIntegerField(default=0, null=False)
    first_date = models.DateField(default=None, null=True)
    last_date = models.DateField(default=None, null=True)
    crs = models.CharField(max_length=10, null=True)

    def __str__(self):
        return self.tile_index


class Zone(models.Model):
    tile = models.ForeignKey(Tile, null=True, on_delete=models.CASCADE, related_name='tile_zones')

    def __str__(self):
        return f"Zone {self.id}"


class Clearcut(models.Model):
    image_date_previous = models.DateField(null=True)
    image_date_current = models.DateField(default=timezone.now)
    area = models.FloatField()
    forest = models.PositiveIntegerField(default=1)
    clouds = models.PositiveIntegerField(default=0)
    centroid = models.PointField(spatial_index=False)
    zone = models.ForeignKey(Zone, on_delete=models.CASCADE)
    mpoly = models.PolygonField(geography=True, srid=4326, spatial_index=True)
    status = models.SmallIntegerField(default=0)
    preview_previous_path = models.URLField(max_length=200, blank=True, null=True, default=None)
    preview_current_path = models.URLField(max_length=200, blank=True, null=True, default=None)

    def __str__(self):
        return f"Clearcut {self.id}"

    # class Meta:
    #     permissions = (
    #         ("can_set_status_as_right", "Set polygon as right predicted"),
    #         ("can_set_status_as_wrong", "Set polygon as wrong predicted")
    #     )


class NotClearcut(models.Model):
    image_date_previous = models.DateField(null=True)
    image_date_current = models.DateField(default=timezone.now)
    area = models.FloatField()
    forest = models.PositiveIntegerField(default=1)
    clouds = models.PositiveIntegerField(default=0)
    centroid = models.PointField(spatial_index=False)
    zone = models.ForeignKey(Zone, on_delete=models.CASCADE)
    mpoly = models.PolygonField(spatial_index=False)

    def __str__(self):
        return f"NotClearcut {self.id}"


class TileInformation(models.Model):
    tile_name = models.CharField(max_length=7, blank=False, null=False)
    tile_index = models.ForeignKey(Tile, on_delete=models.CASCADE, related_name='tile_information')
    tile_date = models.DateField(default=timezone.now)

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
    coordinates = models.PolygonField(blank=True, null=True, spatial_index=False)
    is_downloaded = models.SmallIntegerField(default=0)
    is_prepared = models.SmallIntegerField(default=0)
    is_predicted = models.SmallIntegerField(default=0)
    is_converted = models.SmallIntegerField(default=0)
    is_uploaded = models.SmallIntegerField(default=0)

    objects = models.Manager()


class RunUpdateTask(models.Model):
    tile = models.ForeignKey(Tile, on_delete=models.CASCADE, related_name='run_update_task')
    path_type = models.SmallIntegerField(default=0, null=False)
    path_img_0 = models.URLField(max_length=200, blank=False, null=False, default='')
    path_img_1 = models.URLField(max_length=200, blank=False, null=False, default='')
    image_date_0 = models.DateField(blank=False, null=False)
    image_date_1 = models.DateField(blank=False, null=False)
    path_clouds_0 = models.URLField(max_length=200, blank=False, null=False, default='')
    path_clouds_1 = models.URLField(max_length=200, blank=False, null=False, default='')
    result = models.URLField(max_length=200, null=True)
    date_created = models.DateTimeField(auto_now_add=True)
    date_started = models.DateTimeField(null=True, default=None)
    date_finished = models.DateTimeField(null=True, default=None)

    class Meta:
        db_table = 'clearcuts_run_update_task'
