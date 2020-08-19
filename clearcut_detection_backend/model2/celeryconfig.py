from config import (POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB, DB_HOST,
                    RABBITMQ_USER, RABBITMQ_PASS, RABBITMQ_HOST, RABBITMQ_PORT_NUMBER)

task_serializer = 'json'
result_serializer = 'json'
accept_content = ['json']
# timezone = 'Europe/Oslo'
# enable_utc = True

broker_url = f'amqp://{RABBITMQ_USER}:{RABBITMQ_PASS}@{RABBITMQ_HOST}:{RABBITMQ_PORT_NUMBER}//'
result_backend = f'db+postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{DB_HOST}/{POSTGRES_DB}'

# result_backend = 'rpc://'


# database_table_names = {
#     'task': 'django_celery_results_taskresult',
#     'group': 'myapp_groupmeta',
# }
