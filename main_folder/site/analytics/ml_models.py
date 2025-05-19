# analytics/ml_models.py
import os
import numpy as np
import xgboost as xgb
from core.models import PredictionRequest, PredictionResult
import logging
import traceback
from datetime import datetime
import pandas as pd
from django.conf import settings

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MLModel:
    def __init__(self):
        self.model = None
        self.model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                     'models', 
                                     'xgboost_model.json')
        
    def _load_model(self):
        """Загрузка модели из файла"""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found at {self.model_path}")
            
            if self.model is None:
                model = xgb.Booster()
                model.load_model(self.model_path)
                logger.info(f"Model loaded from {self.model_path}")
                self.model = model
            
            return self.model
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def prepare_features(self, property_data):
        """Подготовка признаков для модели без масштабирования"""
        try:
            logger.info(f"""
            Input property data:
            ID: {property_data.id}
            Latitude: {property_data.latitude}
            Longitude: {property_data.longitude}
            Floor: {property_data.floor}
            Type: {property_data.property_type}
            Class: {property_data.property_class}
            Finishing: {property_data.finishing}
            Purchase date: {property_data.purchase_date}
            """)
            
            # Собираем признаки без масштабирования
            features = [
                float(property_data.latitude),
                float(property_data.longitude),
                float(property_data.floor),
                1.0 if property_data.property_type.lower() == 'квартира' else 0.0,
                1.0 if property_data.finishing == 'С отделки' else 0.0,
                float(self._encode_property_class(property_data.property_class)),
                float(property_data.purchase_date.month),
                float(property_data.purchase_date.year)
            ]
            
            # Создаем массив признаков
            feature_array = np.array([features], dtype=np.float32)
            
            # Логируем значения признаков
            feature_names = ['latitude', 'longitude', 'floor', 'type', 'finishing', 'class', 'month', 'year']
            for name, value in zip(feature_names, features):
                logger.info(f"{name}: {value:.1f}")
            
            return xgb.DMatrix(feature_array)
            
        except Exception as e:
            logger.error(f"Error in prepare_features: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def predict(self, X):
        """Получение прогноза от модели"""
        try:
            if self.model is None:
                self.model = self._load_model()
            
            prediction = self.model.predict(X)
            
            if not isinstance(prediction, np.ndarray) or len(prediction) == 0:
                raise ValueError("Invalid prediction result")
            
            logger.info(f"Raw prediction: {prediction[0]:,.2f}")
            
            return prediction
                
        except Exception as e:
            logger.error(f"Error in predict: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def _encode_property_class(self, property_class):
        """Кодирование класса недвижимости"""
        try:
            mapping = {
                'эконом': 1.0,
                'комфорт': 2.0,
                'комфорт+': 3.0,
                'бизнес': 4.0,
                'элит': 5.0
            }
            
            if isinstance(property_class, (int, float)):
                return float(property_class)
                
            return mapping.get(str(property_class).lower(), 1.0)
            
        except Exception as e:
            logger.error(f"Error encoding property class: {str(e)}")
            return 1.0

def generate_prediction(prediction_request):
    """Генерация прогноза стоимости"""
    try:
        logger.info(f"Starting prediction generation for request: {prediction_request.id}")
        
        model = MLModel()
        property_data = prediction_request.property_data
        
        # Подготовка данных
        X = model.prepare_features(property_data)
        
        # Получение прогноза
        base_price = float(model.predict(X)[0])
        logger.info(f"Base price predicted: {base_price:,.2f}")
        
        # Генерация сценариев с небольшой случайностью
        scenarios = {
            'pessimistic': base_price * (0.9 + np.random.uniform(-0.02, 0.02)),
            'realistic': base_price * (1.0 + np.random.uniform(-0.01, 0.01)),
            'optimistic': base_price * (1.1 + np.random.uniform(-0.02, 0.02))
        }
        
        # Создание результатов
        results = []
        for scenario_type, predicted_price in scenarios.items():
            result = PredictionResult.objects.create(
                prediction_request=prediction_request,
                scenario_type=scenario_type,
                predicted_price=predicted_price,
                influence_factors={
                    'location_attractiveness': 30,
                    'transport_accessibility': 20,
                    'social_infrastructure': 15,
                    'location_development': 20,
                    'macroeconomic': 15
                },
                price_dynamics=_generate_price_dynamics(predicted_price)
            )
            results.append(result)
            logger.info(f"Created {scenario_type} prediction: {predicted_price:,.2f}")
            
        return results
        
    except Exception as e:
        logger.error(f"Error in generate_prediction: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def _generate_price_dynamics(base_price):
    """Генерация динамики цен"""
    try:
        current_date = datetime.now()
        dates = [(current_date.replace(day=1) + pd.DateOffset(months=i)).strftime('%Y-%m')
                for i in range(6)]
        
        # Генерация цен с небольшой случайностью
        base_changes = [0.95, 0.97, 0.99, 1.0, 1.02, 1.04]
        price_changes = [change * (1 + np.random.uniform(-0.01, 0.01)) for change in base_changes]
        prices = [round(base_price * change, 2) for change in price_changes]
        
        logger.info(f"Generated price dynamics: dates={dates}, prices={prices}")
        
        return {
            'dates': dates,
            'prices': prices
        }
    except Exception as e:
        logger.error(f"Error generating price dynamics: {str(e)}")
        return {
            'dates': [],
            'prices': []
        }
