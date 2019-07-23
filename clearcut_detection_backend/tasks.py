import os
import threading

from invoke import task
from os.path import join, splitext

from clearcuts.model_call import call
from clearcuts.geojson_compare import compare
import shutil


def wait_port_is_open(host, port):
    import socket
    import time
    while True:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex((host, port))
            sock.close()
            if result == 0:
                return
        except socket.gaierror:
            pass
        time.sleep(1)


@task
def init_db(ctx, recreate_db=False):
    wait_port_is_open('localhost', 5432)
    if recreate_db:
        pass
        # ctx.run('python manage.py dbshell < clear.sql')
        # ctx.run('python manage.py dbshell < db.dump2404191230')

    ctx.run('python manage.py makemigrations')
    ctx.run('python manage.py migrate')


@task
def collect_static_element(ctx):
    ctx.run('python manage.py collectstatic --noinput')
    ctx.run('python manage.py compilemessages')


@task
def devcron(ctx):
    ctx.run('python devcron.py cron_tab_demo')


@task
def prodcron(ctx):
    ctx.run('python devcron.py cron_tab_prod')


@task
def run(ctx):
    init_db(ctx, recreate_db=False)
    # collect_static_element(ctx)
    # thread_cron = threading.Thread(target=devcron, args=(ctx,))
    # thread_cron.start()
    download_tile(ctx, 'data')
    poly_path, image_path = process_tile(ctx, 'data')
    update_db(ctx, poly_path)
    sendEmail(ctx, image_path)
    # ctx.run('uwsgi --ini uwsgi.ini')


@task
def download_tile(ctx, data_dir):
    ctx.run('python peps_download.py')
    for file in os.listdir(data_dir):
        if file.endswith('.json'):
            os.remove(join(data_dir, file))
        elif file.endswith('.zip'):
            path = join(data_dir, splitext(file)[0])
            ctx.run(f'unzip {path} -d {data_dir}')
            os.remove(path + '.zip')
            ctx.run(f'python prepare_tif.py -f {path}.SAFE')
            shutil.rmtree(f'{path}.SAFE')


@task
def process_tile(ctx, data_dir):
    for file in os.listdir(data_dir):
        if file.endswith('.tif'):
            path = join('..', data_dir, file)
            result_paths = call(path)
            os.remove(join(data_dir, file))

            return result_paths["polygons"], result_paths["picture"]


@task
def update_db(ctx, poly_path):
    compare(poly_path)


@task
def sendEmail(ctx, image_path):
    from django.core.mail.message import EmailMessage

    email = EmailMessage()
    email.subject = "New clearcuts detected"
    email.body = "hi"
    email.from_email = "from@gmail.com"
    email.to = ["to@gmail.com", ]

    email.attach_file(image_path)

    email.send()
    os.remove(image_path)
