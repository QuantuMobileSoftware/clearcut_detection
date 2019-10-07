import os
import shutil
import traceback

from os.path import join, splitext
from clearcuts.geojson_save import save
from model_call import raster_prediction
from django.core.mail.message import EmailMessage
from django.conf import settings


DATA_DIR = 'data'


def download_tile(data_dir):
    # TODO(flying_pi): peps_download.py must be refactored and added main enterpoint.
    # os.system('python peps_download.py')
    import peps_download
    for file in os.listdir(data_dir):
        if file.endswith('.json'):
            os.remove(join(data_dir, file))
        elif file.endswith('.zip'):
            path = join(data_dir, splitext(file)[0])
            os.system(f'unzip {path} -d {data_dir}')
            os.remove(f'{path}.zip')
            os.system(f'python prepare_tif.py -f {path}.SAFE')
            shutil.rmtree(f'{path}.SAFE')


def process_tile(data_dir):
    for file in os.listdir(data_dir):
        if file.endswith('.tif'):
            tiff_path = join(data_dir, file)
            a = 1/0
            result_paths = raster_prediction(tiff_path)[0]
            return result_paths["polygons"], result_paths["picture"]


def init_db(data_dir):
    files = sorted(os.listdir(data_dir))
    for file in files:
        save(os.path.join(data_dir, file), init_db=True)


def update_db(data_dir):
    # TODO download of satellite imagery can be moved to model api in order to reduce amount of times that it is
    # downloaded, but it also means that model instance will spend time downloading data(which is not correct behavior
    # for model instance with gpu)
    download_tile(data_dir)
    poly_path, image_path = process_tile(data_dir)
    save(os.path.join(data_dir, poly_path), init_db=False)


if __name__ == '__main__':
    try:
        # TOOD(flyingpi): Remove or rewrite django stuff after moving it to separate service.
        import django
        django.setup()
        update_db(DATA_DIR)
    except Exception as e:
        EmailMessage(
            subject='Pep download issue',
            body=(
                f'Deamon can not download peps. Issue information listed bellow: '
                f'\n\n{str(e)}\n\n {"".join(traceback.format_tb(e.__traceback__))}'
            ),
            from_email=settings.EMAIL_HOST_USER,
            to=settings.EMAIL_ADMIN_MAILS
        ).send()
