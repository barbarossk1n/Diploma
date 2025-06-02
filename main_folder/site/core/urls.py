# core/urls.py
from django.urls import path
from . import views
from django.contrib.auth.views import LogoutView

urlpatterns = [
    
    path('', views.index, name='index'),
    path('calculate-prediction/', views.calculate_prediction, name='calculate_prediction'),
    path('export-results/<int:prediction_id>/<str:format>/', views.export_results, name='export_results'),
    path('logout/', LogoutView.as_view(), name='logout'),
]
