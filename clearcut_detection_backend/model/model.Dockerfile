FROM python:3.6

RUN mkdir /model

WORKDIR /model

ADD requirements.txt /model

RUN pip install -r requirements.txt

ADD . /model/
