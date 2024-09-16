from config import (POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB, DB_HOST,
                    RABBITMQ_USER, RABBITMQ_PASS, RABBITMQ_HOST, RABBITMQ_PORT_NUMBER)

task_serializer = 'json'
result_serializer = 'json'
accept_content = ['json']

broker_url = f'amqp://{RABBITMQ_USER}:{RABBITMQ_PASS}@{RABBITMQ_HOST}:{RABBITMQ_PORT_NUMBER}//'
result_backend = f'db+postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{DB_HOST}/{POSTGRES_DB}'

worker_prefetch_multiplier = 1
worker_max_tasks_per_child = 1
task_acks_late = True
task_time_limit = 60 * 60
# database_table_names = {
#     'task': 'django_celery_results_taskresult',
#     'group': 'myapp_groupmeta',
# }



