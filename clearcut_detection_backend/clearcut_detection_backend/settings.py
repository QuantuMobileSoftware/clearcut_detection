"""
Django settings for clearcut_detection_backend project.

Generated by 'django-admin startproject' using Django 2.2.3.

For more information on this file, see
https://docs.djangoproject.com/en/2.2/topics/settings/

For the full list of settings and their values, see
https://docs.djangoproject.com/en/2.2/ref/settings/
"""

import os
from pathlib import Path

# Build paths inside the project like this: os.path.join(BASE_DIR, ...)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/2.2/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = 't@wm3&fh5l(w)cb+(9zk%s4r-bmeunos5)&+4)-k2ubxnq+lt4'

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

ALLOWED_HOSTS = ['*']

ROOT_URLCONF = 'clearcut_detection_backend.urls'
WSGI_APPLICATION = 'clearcut_detection_backend.wsgi.application'

# Application definition

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'django.contrib.gis',
    'clearcuts',
    'rest_framework',
    'rest_framework_swagger',
    'corsheaders',
    # 'django_celery_results',
    'downloader.apps.DownloaderConfig',
    'tiff_prepare.apps.TiffPrepareConfig',
]

REST_FRAMEWORK = {
    'DEFAULT_SCHEMA_CLASS': 'rest_framework.schemas.coreapi.AutoSchema',
    'DEFAULT_AUTHENTICATION_CLASSES': (
        'rest_framework_simplejwt.authentication.JWTAuthentication',
    )
}

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
    'corsheaders.middleware.CorsMiddleware'
]

CORS_ORIGIN_ALLOW_ALL = True

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

# Database
# https://docs.djangoproject.com/en/2.2/ref/settings/#databases
# CONN_MAX_AGE = None

DATABASES = {
    'default': {
        'ENGINE': 'django.contrib.gis.db.backends.postgis',
        'NAME': 'clearcuts_db',
        'USER': 'ecoProj',
        'PASSWORD': os.getenv('DB_PASSWORD', 'zys8rwTAC9VIR1X9'),
        'HOST': os.environ.get('DB_HOST', 'db'),
        'PORT': '5432'
    }
}

EMAIL_BACKEND = 'django.core.mail.backends.console.EmailBackend'
EMAIL_HOST = 'smtp.gmail.com'
EMAIL_USE_TLS = True
EMAIL_PORT = 587
EMAIL_HOST_USER = 'from@gmail.com'
EMAIL_HOST_PASSWORD = 'from_password'
EMAIL_ADMIN_MAILS = ['admin@example.com']

# Password validation
# https://docs.djangoproject.com/en/2.2/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]

# In summer can lower cloud percentage to 5-10, amd enlarge nodata pixel threshold to 50 and dates reviewed to 20-25
START_DATE_FOR_SCAN = os.environ.get("START_DATE_FOR_SCAN", "2017-01-01")

MAXIMUM_CLOUD_PERCENTAGE_ALLOWED = 50.0
MAXIMUM_EMPTY_PIXEL_PERCENTAGE = 5.0
MAXIMUM_DATES_REVIEWED_FOR_TILE = 100
MAXIMUM_DATES_STORE_FOR_TILE = 2

MAPBOX_USER = 'quantum-inc'
MAPBOX_ACCESS_TOKEN = os.environ.get('MAPBOX_SECRET_KEY')

# Internationalization
# https://docs.djangoproject.com/en/2.2/topics/i18n/

LANGUAGE_CODE = 'en-us'

TIME_ZONE = 'UTC'

USE_I18N = True

USE_L10N = True

USE_TZ = True


# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/2.2/howto/static-files/

STATIC_URL = '/static/'
STATIC_ROOT = os.path.join(BASE_DIR, "static")

LANDCOVER_URL = 'https://s3-eu-west-1.amazonaws.com\
/vito.landcover.global/2015/E020N60_ProbaV_LC100_epoch2015_global_v2.0.2_products_EPSG-4326.zip'

MAX_WORKERS = 6

DATA_DIR = Path('data')
DOWNLOADED_IMAGES_DIR = Path('./data/source_images/')
MODEL_TIFFS_DIR = Path('./data/model_tiffs')
MAPBOX_TIFFS_DIR = Path('./data/mapbox_tiffs')

PATH_TYPE = 'fs'

# Celery settings
RABBITMQ_USER = os.environ.get('RABBITMQ_DEFAULT_USER', 'guest')
RABBITMQ_PASS = os.environ.get('RABBITMQ_DEFAULT_PASS', 'guest')
RABBITMQ_HOST = os.environ.get('RABBITMQ_HOST', 'rabbitmq_prod')
RABBITMQ_PORT_NUMBER = os.environ.get('RABBITMQ_NODE_PORT_NUMBER', 5672)

CELERY_BROKER_URL = f'amqp://{RABBITMQ_USER}:{RABBITMQ_PASS}@{RABBITMQ_HOST}:{RABBITMQ_PORT_NUMBER}//'
CELERY_ACCEPT_CONTENT = ['json']
CELERY_TASK_SERIALIZER = 'json'
CELERY_RESULT_SERIALIZER = 'json'

CELERY_RESULT_BACKEND = f'db+postgresql://{DATABASES["default"]["USER"]}:{os.getenv("DB_PASSWORD", "zys8rwTAC9VIR1X9")}\
@{os.environ.get("DB_HOST", "db")}/{DATABASES["default"]["NAME"]}'


LOGGING = {
        'version': 1,
    'disable_existing_loggers': False,
    'handlers': {
        'file': {
            'level': 'WARNING',
            'class': 'logging.FileHandler',
            'filename': 'WARNING-ERROR.log',
        },
        'console': {
            'class': 'logging.StreamHandler',
        },
        'mail_on_error': {
            'level': 'WARNING',
            'class': 'services.email_on_error.EmailOnErrorHandler',
        }
    },
    'loggers': {
        'update': {
            'handlers': ['file', 'console', 'mail_on_error'],
            'propagate': False,
            'level': os.getenv('DJANGO_LOG_LEVEL', 'INFO')
        },
        'sentinel': {
            'handlers': ['file', 'console', 'mail_on_error'],
            'propagate': False,
            'level': os.getenv('DJANGO_LOG_LEVEL', 'INFO')
        }, 
        'prepare_tif': {
            'handlers': ['file', 'console', 'mail_on_error'],
            'propagate': False,
            'level': os.getenv('DJANGO_LOG_LEVEL', 'INFO')
        }, 
        'jp2_to_tiff_conversion': {
            'handlers': ['file', 'console', 'mail_on_error'],
            'propagate': False,
            'level': os.getenv('DJANGO_LOG_LEVEL', 'INFO')
        },
        'model_call': {
            'handlers': ['file', 'console', 'mail_on_error'],
            'propagate': False,
            'level': os.getenv('DJANGO_LOG_LEVEL', 'INFO')
        },
        'create_update_task': {
            'handlers': ['file', 'console', 'mail_on_error'],
            'propagate': False,
            'level': os.getenv('DJANGO_LOG_LEVEL', 'INFO')
        },         
        'img_preprocessing': {
            'handlers': ['file', 'console', 'mail_on_error'],
            'propagate': False,
            'level': os.getenv('DJANGO_LOG_LEVEL', 'INFO')
        },
    },
}
