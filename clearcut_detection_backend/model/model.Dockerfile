FROM nvidia/cuda:10.0-cudnn7-runtime-ubuntu18.04

RUN apt-get update && apt-get install -y python3-pip

RUN mkdir /model

WORKDIR /model

ADD requirements.txt /model

RUN pip3 install -r requirements.txt

ADD . /model/
