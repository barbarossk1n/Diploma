# api/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('calculate-prediction/', views.calculate_prediction, name='api_calculate_prediction'),
    path('prediction-results/<int:prediction_id>/', views.get_prediction_results, name='api_prediction_results'),
    path('property-info/<int:property_id>/', views.get_property_info, name='api_property_info'),
    path('auth/check/', views.check_auth, name='api_auth_check'),
]
