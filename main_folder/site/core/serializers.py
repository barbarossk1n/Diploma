# core/serializers.py
from rest_framework import serializers
from .models import PredictionResult

class PredictionResultSerializer(serializers.ModelSerializer):
    class Meta:
        model = PredictionResult
        fields = ['id', 'scenario_type', 'predicted_price', 'influence_factors', 
                 'price_dynamics', 'created_at']

    def to_representation(self, instance):
        data = super().to_representation(instance)
        data['predicted_price'] = float(data['predicted_price'])
        return data
