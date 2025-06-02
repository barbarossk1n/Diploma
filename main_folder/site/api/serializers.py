# api/serializers.py
# api/serializers.py
from rest_framework import serializers
from core.models import Property, PredictionRequest, PredictionResult

class PropertySerializer(serializers.ModelSerializer):
    class Meta:
        model = Property
        fields = [
            'id',
            'complex_name',
            'location_address',
            'region',
            'latitude',
            'longitude',
            'floor',
            'total_floors',
            'area',
            'property_type',
            'finishing',
            'property_class',
            'build_year',
            'purchase_date',
            'created_at',
            'updated_at'
        ]

class PredictionRequestSerializer(serializers.ModelSerializer):
    property_data = PropertySerializer(read_only=True)
    
    class Meta:
        model = PredictionRequest
        fields = [
            'id',
            'user',
            'property_data',
            'investment_strategy',
            'inflation_rate',
            'central_bank_rate',
            'consumer_price_index',
            'gdp_growth_rate',
            'mortgage_rate',
            'deposit_rate',
            'created_at'
        ]

class PredictionResultSerializer(serializers.ModelSerializer):
    class Meta:
        model = PredictionResult
        fields = [
            'id',
            'prediction_request',
            'scenario_type',
            'predicted_price',
            'influence_factors',
            'price_dynamics',
            'created_at'
        ]

    def to_representation(self, instance):
        """Преобразование данных для JSON-сериализации"""
        data = super().to_representation(instance)
        # Преобразуем Decimal в float для predicted_price
        data['predicted_price'] = float(data['predicted_price'])
        return data

