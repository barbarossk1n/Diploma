# api/serializers.py
from rest_framework import serializers
from core.models import Property, PredictionRequest, PredictionResult

class PropertySerializer(serializers.ModelSerializer):
    class Meta:
        model = Property
        fields = '__all__'

class PredictionRequestSerializer(serializers.ModelSerializer):
    property_data = PropertySerializer(read_only=True)
    
    class Meta:
        model = PredictionRequest
        fields = '__all__'

class PredictionResultSerializer(serializers.ModelSerializer):
    class Meta:
        model = PredictionResult
        fields = '__all__'
