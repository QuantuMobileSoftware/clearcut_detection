FROM python:3.7.7

ENV PYTHONUNBUFFERED 1
ENV ROOTDIR /usr/local/
ARG GDAL_VERSION=3.1.4

WORKDIR ${ROOTDIR}/

RUN apt-get update && apt-get upgrade -y\
    && apt-get install software-properties-common -y\
    && add-apt-repository "deb http://apt.postgresql.org/pub/repos/apt/ $(lsb_release -sc)-pgdg main"\
    && wget --quiet -O - https://www.postgresql.org/media/keys/ACCC4CF8.asc | apt-key add -\
    && apt-get update\
    && apt-get install postgresql-client-13 -y\
    && apt-get install sqlite3 -y

# See https://docs.djangoproject.com/en/3.1/ref/contrib/gis/install/geolibs/

RUN wget https://download.osgeo.org/geos/geos-3.8.1.tar.bz2 \
    && tar -xjf geos-3.8.1.tar.bz2 \
    && cd geos-3.8.1 \
    && ./configure \
    && make \
    && make install \
    && cd .. \
    && rm -rf geos-3.8.1 geos-3.8.1.tar.bz2

RUN wget https://download.osgeo.org/proj/proj-6.3.2.tar.gz \
    && wget https://download.osgeo.org/proj/proj-datumgrid-1.8.tar.gz \
    && tar -xzf proj-6.3.2.tar.gz \
    && mkdir proj-6.3.2/nad && cd proj-6.3.2/nad \
    && tar -xzf ../../proj-datumgrid-1.8.tar.gz \
    && cd .. \
    && ./configure \
    && make \
    && make install \
    && cd .. \
    && rm -rf proj-6.3.2 proj-6.3.2.tar.gz proj-datumgrid-1.8.tar.gz

RUN apt-get -y install python3-pip
RUN pip3 install numpy==1.17.3

RUN wget https://download.osgeo.org/gdal/${GDAL_VERSION}/gdal-${GDAL_VERSION}.tar.gz \
    && tar -xzf gdal-${GDAL_VERSION}.tar.gz \
    && cd gdal-${GDAL_VERSION} \
    && ./configure --with-python=python3.7 --with-proj=${ROOTDIR} \
    && make \
    && make install \
    && cd swig/python \
    && python3 setup.py build \
    && python3 setup.py install \
    && cd ${ROOTDIR} \
    && rm -rf gdal-${GDAL_VERSION} gdal-${GDAL_VERSION}.tar.gz

RUN git clone https://github.com/mapbox/tippecanoe.git \
    && cd tippecanoe \
    && git checkout tags/1.36.0 \
    && make -j \
    && make install \
    && cd .. \
    && rm -rf tippecanoe

RUN ldconfig

RUN mkdir /code

WORKDIR /code

COPY requirements.txt /code/

RUN pip3 install -r requirements.txt