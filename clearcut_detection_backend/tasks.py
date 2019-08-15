import os
import threading

from invoke import task


@task
def run(ctx):
    init_db(ctx, create_db=False)
    ctx.run('python update.py')


@task
def runcron(ctx):
    init_db(ctx, create_db=False)
    thread_cron = threading.Thread(target=devcron, args=(ctx,))
    thread_cron.start()


@task
def devcron(ctx):
    ctx.run('python devcron.py cron_tab_prod')


@task
def collect_static_element(ctx):
    ctx.run('python manage.py collectstatic --noinput')


@task
def init_db(ctx, create_db=False):
    wait_port_is_open(os.getenv('DB_HOST', 'db'), 5432)
    if create_db:
        ctx.run('python manage.py loaddata db.json')

    ctx.run('python manage.py makemigrations clearcuts')
    ctx.run('python manage.py migrate')


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
def rundev(ctx, createdb=False):
    init_db(ctx, createdb)
    collect_static_element(ctx)
    ctx.run('uwsgi --ini uwsgi.ini')


@task
def runbackend(ctx, createdb=False):
    init_db(ctx, createdb)
    collect_static_element(ctx)
    thread_cron = threading.Thread(target=devcron, args=(ctx,))
    thread_cron.start()
    ctx.run('uwsgi --ini uwsgi.ini')
