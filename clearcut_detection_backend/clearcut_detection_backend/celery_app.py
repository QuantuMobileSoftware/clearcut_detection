import os
from celery import Celery
from celery.backends.database import SessionManager

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'clearcut_detection_backend.settings')

app = Celery('clearcut_detection_backend')

app.config_from_object('django.conf:settings', namespace='CELERY')

session = SessionManager()
engine = session.get_engine(app.backend.url)
session.prepare_models(engine)

app.autodiscover_tasks()
