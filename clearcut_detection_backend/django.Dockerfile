FROM python:3.6

RUN mkdir /code

WORKDIR /code

RUN apt-get update -y \
    && apt-get upgrade -y \
    && apt-get install -y postgresql-client locales \
    && apt-get install -y gdal-bin python-gdal python3-gdal \
    && apt-get update && apt-get install -y gettext libgettextpo-dev \
    # Cleanup
    && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

ADD requirements.txt /code

RUN pip install -r requirements.txt

RUN mkdir /code/backend
ADD clearcut_detection_backend /code/backend/

RUN mkdir /code/pytorch
ADD pytorch /code/pytorch/

WORKDIR /code/backend