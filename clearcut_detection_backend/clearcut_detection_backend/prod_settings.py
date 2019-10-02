from clearcut_detection_backend.settings import *

DEBUG = False

EMAIL_BACKEND = 'django.core.mail.backends.smtp.EmailBackend'
EMAIL_HOST = 'smtp.gmail.com'
EMAIL_USE_TLS = True
EMAIL_PORT = 587
EMAIL_HOST_USER = 'SET_ME_PLEASE_SMTP_USERNAME'
EMAIL_HOST_PASSWORD = 'SET_ME_PLEASE_SMTP_PASSWORD'
EMAIL_ADMIN_MAIL = ['SET_ME_PLEASE_SMTP_ADMIN_MAIL']
