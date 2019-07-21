import os
import threading

from invoke import task
from os.path import join, splitext

from clearcuts.model_call import call

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
        ctx.run('python manage.py dbshell < clear.sql')
        # ctx.run('python manage.py dbshell < db.dump2404191230')

    # ctx.run('python manage.py makemigrations')
    # ctx.run('python manage.py migrate')


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
    # init_db(ctx, recreate_db=True)
    # collect_static_element(ctx)
    # thread_cron = threading.Thread(target=devcron, args=(ctx,))
    # thread_cron.start()
    # download_tile(ctx, 'data')
    process_tile(ctx, 'data')
    # ctx.run('uwsgi --ini uwsgi.ini')


@task
def run_demo(ctx):
    init_db(ctx)
    collect_static_element(ctx)
    thread_cron = threading.Thread(target=devcron, args=(ctx,))
    thread_cron.start()
    ctx.run('uwsgi --ini uwsgi.ini')


@task
def run_prod(ctx):
    ctx.run('python manage.py migrate waffle')
    ctx.run('python manage.py migrate --noinput')
    # TODO temp workaround to alter migrations that are made by
    #  allauth.account to remove email unique constaints
    ctx.run('python manage.py dbshell < drop_constraint.sql')
    thread_cron = threading.Thread(target=prodcron, args=(ctx,))
    thread_cron.start()
    ctx.run('uwsgi --ini uwsgi.ini')


@task
def download_tile(ctx, data_dir):
    ctx.run('python peps_download.py')
    for file in os.listdir(data_dir):
        if file.endswith('.zip'):
            path = join(data_dir, splitext(file)[0])
            ctx.run(f'unzip {path} -d {data_dir}')
            os.remove(path + '.zip')
            ctx.run(f'python prepare_tif.py -f {path}.SAFE')
            os.rmdir(f'{path}.SAFE')

@task
def process_tile(ctx, data_dir):
    for file in os.listdir(data_dir):
        if file.endswith('.tif'):
            path = join('../clearcut_detection_backend', data_dir, file)
            call(path)