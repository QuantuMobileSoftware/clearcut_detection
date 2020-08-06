"""
Save geojson tio db
"""
import sys
import logging
from distutils.util import strtobool
import django
django.setup()
from clearcuts.geojson_save import save_from_task


logger = logging.getLogger('update')


if __name__ == '__main__':
    print(sys.argv[1:])
    task_id = int(sys.argv[1])
    save_from_task(task_id)
