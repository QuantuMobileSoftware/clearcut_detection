Launch requirements:
Run `docker-compose -f docker-compose.dev.yml up` in order to run docker for backend and frontend development.
Change `docker-compose.dev.yml` `command: /bin/bash -c "pip install -r requirements.txt && exec invoke rundev"` to `command: /bin/bash -c "pip install -r requirements.txt && exec invoke runbackend"` to run both backend server and background processing. 
