FROM python:3.6

RUN apt-get update -y \
    && apt-get upgrade -y \
    && apt-get install -y postgresql-client locales \
    && apt-get install -y gdal-bin python-gdal python3-gdal \
    && apt-get update && apt-get install -y gettext libgettextpo-dev \
    # Cleanup
    && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

COPY requirements.txt /tmp
RUN pip install -r /tmp/requirements.txt
