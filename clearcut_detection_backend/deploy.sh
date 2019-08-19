#!/bin/bash
docker-compose -f docker-compose-stage.yml up react
docker-compose -f docker-compose-stage.yml up -d
cp ./frontend/build/* ./static
cp -R ./frontend/build/static/* ./static
