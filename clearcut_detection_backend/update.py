import os
import shutil

from os.path import join, splitext
from clearcuts.geojson_save import save
from model_call import raster_prediction
from django.core.mail.message import EmailMessage


def download_tile(data_dir):
    os.system('python peps_download.py')
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
            result_paths = raster_prediction(tiff_path)[0]
            return result_paths["polygons"], result_paths["picture"]


def send_email(image_path):
    email = EmailMessage()
    email.subject = "New clearcuts detected"
    email.body = "hi"
    email.from_email = "from@gmail.com"
    email.to = ["to@gmail.com", ]

    email.attach_file(image_path)
    email.send()
    os.remove(image_path)


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
    DATA_DIR = '/data'
    update_db(DATA_DIR)
