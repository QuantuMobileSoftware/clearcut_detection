#!/usr/bin/env bash

cd ..
docker-compose -f ./docker-compose-for-django-dev.yml run django python /code/manage.py createsuperuser
# TODO find how to remove container after job done.
#docker-compose -f ./docker-compose-for-django-dev.yml stop django && docker-compose rm -f django

