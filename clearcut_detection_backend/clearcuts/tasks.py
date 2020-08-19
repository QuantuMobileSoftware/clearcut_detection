import django
django.setup()
from clearcut_detection_backend import app
from clearcuts.geojson_save import save_from_task as save_polygons


@app.task(name='tasks.save_from_task')
def save_from_task(task_id):
    save_polygons(task_id)
