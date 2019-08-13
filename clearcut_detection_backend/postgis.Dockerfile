FROM mdillon/postgis:9.6
MAINTAINER Konstantin Bondarenko <k.bondarenko@quantumobile.com>

ENV PGROUTING_MAJOR 2.3
ENV PGROUTING_VERSION 2.3.0-1
#ToDo: version control
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      wget \
      postgresql-$PG_MAJOR-pgrouting && \
    rm -rf /var/lib/apt/lists/*
