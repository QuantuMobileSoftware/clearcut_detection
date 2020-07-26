#!/bin/bash

celery -A tasks worker -Q model_predict_queue --loglevel=DEBUG
