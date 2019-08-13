FROM python:3.6

RUN mkdir /model

WORKDIR /model

COPY model_requirements.txt /model

RUN pip install -r model_requirements.txt

COPY . /model/
