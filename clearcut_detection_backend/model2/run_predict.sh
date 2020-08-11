#!/bin/bash

celery -A tasks worker -Q model_predict_queue --concurrency=2 --loglevel=DEBUG
