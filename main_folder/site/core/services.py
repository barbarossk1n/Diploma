# core/services.py

# core/services.py
import logging
from .models import PredictionResult
from .models.ml_model import MLModel
from django.core.exceptions import ValidationError

logger = logging.getLogger(__name__)

def prepare_features(prediction_request):
    """
    Подготовка признаков для модели из запроса
    
    Args:
        prediction_request (PredictionRequest): Объект запроса на прогноз

    Returns:
        array-like: Подготовленные признаки для модели
    """
    try:
        # Здесь должна быть логика подготовки признаков
        # Это пример, нужно адаптировать под реальные данные
        features = [
            prediction_request.location_data,
            prediction_request.transport_data,
            prediction_request.social_data,
            prediction_request.development_data,
            prediction_request.macro_data
        ]
        return features
    except Exception as e:
        logger.error(f"Error preparing features: {str(e)}")
        raise ValueError("Failed to prepare features for prediction")

def format_prediction_results(prediction, prediction_request):
    """
    Форматирование результатов предсказания в требуемый формат
    
    Args:
        prediction: Результат предсказания модели
        prediction_request (PredictionRequest): Исходный запрос

    Returns:
        list: Список объектов PredictionResult
    """
    try:
        results = []
        
        # Реалистичный сценарий
        realistic_result = PredictionResult.objects.create(
            prediction_request=prediction_request,
            scenario_type='realistic',
            predicted_price=prediction,  # Используем предсказание модели
            influence_factors={
                'location': 30,
                'transport': 20,
                'social': 15,
                'development': 20,
                'macro': 15
            },
            price_dynamics={
                'dates': ['2025-01', '2025-02', '2025-03', '2025-04'],
                'prices': [prediction * 0.97, prediction * 0.99, prediction, prediction * 1.02]
            }
        )
        results.append(realistic_result)
        
        # Оптимистичный сценарий
        optimistic_result = PredictionResult.objects.create(
            prediction_request=prediction_request,
            scenario_type='optimistic',
            predicted_price=prediction * 1.1,  # +10% к базовому предсказанию
            influence_factors={
                'location': 25,
                'transport': 25,
                'social': 15,
                'development': 20,
                'macro': 15
            },
            price_dynamics={
                'dates': ['2025-01', '2025-02', '2025-03', '2025-04'],
                'prices': [prediction, prediction * 1.03, prediction * 1.06, prediction * 1.1]
            }
        )
        results.append(optimistic_result)
        
        # Пессимистичный сценарий
        pessimistic_result = PredictionResult.objects.create(
            prediction_request=prediction_request,
            scenario_type='pessimistic',
            predicted_price=prediction * 0.9,  # -10% к базовому предсказанию
            influence_factors={
                'location': 35,
                'transport': 15,
                'social': 15,
                'development': 20,
                'macro': 15
            },
            price_dynamics={
                'dates': ['2025-01', '2025-02', '2025-03', '2025-04'],
                'prices': [prediction, prediction * 0.97, prediction * 0.93, prediction * 0.9]
            }
        )
        results.append(pessimistic_result)
        
        return results
        
    except Exception as e:
        logger.error(f"Error formatting prediction results: {str(e)}")
        raise

def generate_prediction(prediction_request):
    """
    Генерация прогноза на основе запроса
    
    Args:
        prediction_request (PredictionRequest): Объект запроса на прогноз

    Returns:
        list: Список результатов прогноза для разных сценариев
        
    Raises:
        ValueError: При ошибке в подготовке данных
        Exception: При других ошибках в процессе генерации прогноза
    """
    try:
        # Инициализация модели
        model = MLModel()
        if not model:
            raise ValueError("Failed to initialize prediction model")
            
        # Подготовка признаков
        features = prepare_features(prediction_request)
        if not features:
            raise ValueError("Failed to prepare features for prediction")
            
        # Получение предсказания
        prediction = model.predict(features)
        if prediction is None:
            raise ValueError("Model failed to generate prediction")
            
        # Форматирование результатов
        results = format_prediction_results(prediction, prediction_request)
        if not results:
            raise ValueError("Failed to format prediction results")
            
        return results
        
    except ValueError as e:
        logger.error(f"Validation error in generate_prediction: {str(e)}")
        raise
        
    except Exception as e:
        logger.error(f"Error in generate_prediction: {str(e)}")
        raise
