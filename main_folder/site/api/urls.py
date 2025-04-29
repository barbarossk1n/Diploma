# api/urls.py
from django.urls import path
from . import views

urlpatterns = [
    # API endpoints будут добавлены позже
    path('user-predictions/', views.user_predictions, name='user_predictions'),
    path('prediction-detail/<int:prediction_id>/', views.prediction_detail, name='prediction_detail'),
]
