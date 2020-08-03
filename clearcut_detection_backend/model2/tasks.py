import os
from celery import Celery
from config import db_string
from db_engin import make_session_factory
from run_predict_tasks.run_predict import run_predict

broker = 'amqp://guest:guest@rabbitmq:5672//'  # TODO
backend = f'db+postgresql://{os.environ.get("POSTGRES_USER", "ecoProj")}:{os.getenv("DB_PASSWORD", "zys8rwTAC9VIR1X9")}\
@{os.environ.get("DB_HOST", "db")}/{os.environ.get("POSTGRES_DB", "clearcuts_db")}'

app = Celery(
    'model',
    broker=broker,
    backend=backend
)


@app.task(remote_tracebacks='enable')
def run_model_predict(**kwargs):
    task_id = int(kwargs.get('task_id'))
    print(f'get task_id: {task_id}')
    session_factory = make_session_factory(db_string)
    session = session_factory.make_session()
    json_file = run_predict(session, task_id)
    print(f'job done, all poligons in: {json_file}')
    return task_id
