#!/bin/bash

celery -A tasks worker -Q model_predict_queue --concurrency=8 --loglevel=DEBUG
