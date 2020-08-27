from django.urls import path
from rest_framework.urlpatterns import format_suffix_patterns
from clearcuts import views

urlpatterns = [
    path('info/<start_date>/<end_date>', views.clearcuts_info),
    path('area_chart/<int:id>/<start_date>/<end_date>', views.clearcut_area_chart),
    path('<int:pk>', views.ClearcutDetail.as_view()),
    path('run_update_task', views.RunUpdateTaskList.as_view()),
]

urlpatterns = format_suffix_patterns(urlpatterns)
