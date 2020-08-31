import time
from celery import Celery, Task
from celery.exceptions import TimeLimitExceeded
from config import (POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB, DB_HOST)
from db_engin import make_session_factory
from run_predict_tasks.run_predict import run_predict


db_string = f'postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{DB_HOST}/{POSTGRES_DB}'

app = Celery('model',)
app.config_from_object('celeryconfig')

# https://docs.celeryproject.org/en/stable/userguide/configuration.html#configuration


class CallbackTask(Task):
    def on_success(self, retval, task_id, args, kwargs):
        pass
        # app.send_task(
        #     'tasks.save_from_task',
        #     queue='save_from_task_queue',
        #     args=(retval, ),
        # )

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        return 'by'


# @app.task(remote_tracebacks='enable', base=CallbackTask) soft_time_limit=15 * 60, time_limit=15*60
@app.task(remote_tracebacks='enable', autoretry_for=(TimeLimitExceeded,), max_retries=2)
def run_model_predict(**kwargs):
    task_id = int(kwargs.get('task_id'))
    print(f'get task_id: {task_id}')
    start_time = time.time()
    session_factory = make_session_factory(db_string)
    session = session_factory.make_session()
    json_file = run_predict(session, task_id)
    print(f'job done, all poligons in: {json_file}')
    # time.sleep(task_id-840)
    print(f'---{time.time() - start_time} seconds --- for predicting task_id: {task_id}')
    return task_id
