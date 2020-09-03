# Clearcut_detection

## Project structure info
 * `clearcut_detection_backend` - web-service for clearcut detection
 * `clearcut_research` - investigation about model approach, model training and model evaluation of clearcut detection
 
## Launch requirements:  
To start a web-service, do next:
* `cd clearcut_detection_backend/`  
* create db.env based on the db.env.example
* create `gcp_config.ini` based on `gcp_config.ini.dist`
        `AREA_TILE_SET` value is responsible for region that should be fetched and processedUpdate it if needed. 
        To get tiles Ids you can use https://mappingsupport.com/p2/gissurfer.php?center=14SQH05239974&zoom=4&basemap=USA_basemap
          
* create key.json based on the key.json.example
        In order to use Google Cloud Storage you need to generate service account key and put this file inside /clercut_detection_backend folder, 
        file name should be specified inside gcp_config file(key.json by default).
        for more information about creating key.json read https://flaviocopes.com/google-api-authentication/
        
* create django.env based on the django.env.dist
* create rabbitmq.env based on the rabbitmq.env.dist

* `cd model2/` now You should be in `clearcut_detection_backend/model2/`
* create model.env file based on model.env.dist,
    RABBITMQ_DEFAULT_USER and RABBITMQ_DEFAULT_PASS copy from rabbitmq.env created earlier, 
    POSTGRES_USER and POSTGRES_PASSWORD copy from db.env created earlier
* put same key.json  generated earlier to clearcut_detection_backend/model2/key.json
* put unet_v4.pth in to  clearcut_detection_backend/model2/unet_v4.pth (trained model can be obtained from maintainers)
* Run `docker build -f model.Dockerfile -t clearcut_detection/model2 .` in order to build model2 docker image

## before start
* `cd ../clearcut_detection_backend` now You should be in `clearcut_detection_backend/clearcut_detection_backend/`
* In `settings.py` edit `MAX_WORKERS` - depends of your CPU cores number
* In `prod_settings.py` edit email settings 
* `cd ..` now You should be in `clearcut_detection_backend/`
* In `docker-compose-stage.yml` edit:
    * `CUDA_VISIBLE_DEVICES` - set cuda device You Want to use, default -0
    * `celery -A tasks worker -Q model_predict_queue --concurrency=2` - set concurrency=x where x number of celery workers
    * `VIRTUAL_HOST`
    * `VIRTUAL_PORT`
    * `LETSENCRYPT_HOST`
    * `LETSENCRYPT_EMAIL`

## At the first start:
* Run `docker build -f postgis.Dockerfile -t clearcut_detection/postgis .` in order to build postgis docker image
* Run `docker build -f django.Dockerfile -t clearcut_detection/backend .` in order to build backend image
* Run `docker-compose -f docker-compose-stage.yml up -d db_stage` in order to run docker for data base
* Run `docker-compose -f ./docker-compose-stage.yml run --rm django_stage python /code/manage.py migrate --noinput`
    in order to create all data base tables
* Run `docker-compose -f ./docker-compose-stage.yml run --rm django_stage python /code/manage.py loaddata db.json`
    in order to import data from db.json file to data base (if You want to use our demo data base, dont run it if You dont need information about demo regions)
* Run `docker-compose -f ./docker-compose-stage.yml run --rm django_stage python /code/manage.py createsuperuser` in order to create django superuser
    
## Launch project
* Run `docker-compose -f docker-compose-stage.yml up` for deployment.

## Collect new data from scratch
* In `docker-compose-stage.yml` edit: `START_DATE_FOR_SCAN` start date for scan images
* Run `docker-compose -f ./docker-compose-stage.yml up -d model_stage_2` - in order to run prediction model
* Run `docker-compose -f ./docker-compose-stage.yml run django_stage python /code/update_all.py --exit-code-from django_stage --abort-on-container-exit django_stage` 
    - in order to run script for collecting images for all tiles from `gcp_config.ini`
    
## Fetch new data
Use if You have already collected data, and now You want to update it by new information
* Run `docker-compose -f ./docker-compose-stage.yml up -d model_stage_2` - in order to run or rerun prediction model
* Run `docker-compose -f ./docker-compose-stage.yml run django_stage python /code/update_new.py --exit-code-from django_stage --abort-on-container-exit django_stage`
    - in order to run script for collecting new images for all staged tiles from data base
    
## Swagger:  
After the app has been launched, swagger for api can be used. Go to http://localhost/api/swagger to access swagger with full description of api endpoints.
You have to login as superuser to see all API description.

## Whats new
https://github.com/QuantuMobileSoftware/clearcut_detection/blob/master/clearcut_detection_backend/docs/Whats_new
