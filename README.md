# Clearcut_detection  

## Project structure info
 * `clearcut_detection_backend` - web-service for clearcut detection
 * `clearcut_research` - investigation about model approach, model training and model evaluation of clearcut detection
 
 ## Launch requirements:  
To start a wev-service, do next:
* `cd clearcut_detection_backend/`  
* Run `docker-compose -f docker-compose.dev.yml up` in order to run docker for backend and frontend development.  
* Run `docker-compose -f docker-compose-stage.yml up` for deployment.

## Swagger:  
After the app has been launched, swagger for api can be used. Go to http://localhost/api/swagger to access swagger with full description of api endpoints.
