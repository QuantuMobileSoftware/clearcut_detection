import os
import sys
import shutil
import argparse

from os.path import join, splitext

def download_tile(data_dir, save_dir, no_download):
    if not no_download:
        os.system('python peps_download.py')
    for file in os.listdir(data_dir):
        if file.endswith('.json'):
            os.remove(join(data_dir, file))
        elif file.endswith('.zip'):
            path = join(data_dir, splitext(file)[0])
            os.system(f'unzip {path} -d {data_dir}')
            if not no_download:
                os.remove(f'{path}.zip')
            save_path = f'{data_dir}/input'
            os.system(f'python prepare_tif.py -f {path}.SAFE -s {save_dir}')
            shutil.rmtree(f'{path}.SAFE')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Script for downloading images.')
    parser.add_argument(
        '--data_dir', '-d', dest='data_dir',
        required=True, help='Path to the directory where images are stored.'
    )
    parser.add_argument(
        '--save_dir', '-s', dest='save_dir',
        required=True, help='Path to the directory where images will be saved.'
    )
    parser.add_argument(
        "-n", "--no_download", dest="no_download", action="store_true",
        help="Do not download products, just prepare tiff",
        default=False
    )
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    download_tile(args.data_dir, args.save_dir, args.no_download)