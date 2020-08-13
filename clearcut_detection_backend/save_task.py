"""
Save geojson tio db
"""
import sys
import logging
from distutils.util import strtobool
import django
django.setup()
from clearcuts.geojson_save import save_from_task
from clearcuts.models import RunUpdateTask


logger = logging.getLogger('update')


if __name__ == '__main__':

    tasks = RunUpdateTask.objects.filter(result__isnull=False)

    for task in tasks:
        task_id = task.id
        save_from_task(task_id)
