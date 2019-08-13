"""clearcut_detection_backend URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import include, path
from clearcuts import views


from rest_framework_swagger.views import get_swagger_view

schema_view = get_swagger_view(title='Clearcut detection API')

urlpatterns = [
    path('api/swagger', schema_view),
    path('api/clearcuts_info/<start_date>/<end_date>', views.clearcuts_info),
    path('api/clearcut_area_chart/<int:id>/<start_date>/<end_date>', views.clearcut_area_chart),
    path('admin/', admin.site.urls),
]
