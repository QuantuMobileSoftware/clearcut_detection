from django.contrib.gis.utils import LayerMapping

from .models import Clearcut

clearcuts_mapping = {
    'forest_type': 'forest_typ',
    'forest_state': 'state',
    'detected_class': 'type',
    'image_date': 'img_date',
    'mpoly': 'POLYGON',
}


def run(clearcuts_shp, verbose=True):
    lm = LayerMapping(Clearcut, clearcuts_shp, clearcuts_mapping, transform=False)
    lm.save(strict=True, verbose=verbose)
