import os


def path_exists_or_create(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path