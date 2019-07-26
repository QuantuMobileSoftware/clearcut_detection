import os
import shutil

from os.path import join, splitext
from clearcuts.model_call import call
from clearcuts.geojson_compare import compare
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
            path = join('..', data_dir, file)
            result_paths = call(path)
            os.remove(join(data_dir, file))

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


if __name__ == '__main__':
    download_tile('data')
    poly_path, image_path = process_tile('data')
    compare(poly_path)
    send_email(image_path)