FROM python:3.6

RUN mkdir /test
WORKDIR /test

COPY requirements.txt /test
RUN pip install -r requirements.txt

RUN apt-get update -y && apt-get install -y \
    software-properties-common

RUN add-apt-repository -r ppa:ubuntugis/ppa && apt-get update
RUN apt-get update
RUN apt-get install gdal-bin -y
RUN apt-get install libgdal-dev -y
RUN export CPLUS_INCLUDE_PATH=/usr/include/gdal
RUN export C_INCLUDE_PATH=/usr/include/gdal
RUN pip install GDAL==$(gdal-config --version | awk -F'[.]' '{print $1"."$2}')


ADD . /test