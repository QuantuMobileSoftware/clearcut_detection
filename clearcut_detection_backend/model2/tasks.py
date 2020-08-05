from celery import Celery
from config import (POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB, DB_HOST,
                    RABBITMQ_USER, RABBITMQ_PASS, RABBITMQ_HOST, RABBITMQ_PORT_NUMBER)
from db_engin import make_session_factory
from run_predict_tasks.run_predict import run_predict


broker = f'amqp://{RABBITMQ_USER}:{RABBITMQ_PASS}@{RABBITMQ_HOST}:{RABBITMQ_PORT_NUMBER}//'
backend = f'db+postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{DB_HOST}/{POSTGRES_DB}'
db_string = f'postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{DB_HOST}/{POSTGRES_DB}'

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
