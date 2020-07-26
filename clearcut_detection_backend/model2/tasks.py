from celery import Celery
from config import db_string
from db_engin import make_session_factory
from run_predict_tasks.run_predict import run_predict

# broker = config.get('celery', 'broker')
broker = 'amqp://guest:guest@rabbitmq:5672//'  # TODO
app = Celery(
    'backtest',
    broker=broker
)


@app.task(remote_tracebacks='enable')
def run_model_predict(**kwargs):
    task_id = int(kwargs.get('task_id'))
    print(f'qqqqqqqqqqqqqqqqqqqqqqq {task_id}')
    session_factory = make_session_factory(db_string)
    session = session_factory.make_session()
    run_predict(session, task_id)
