# api/views.py
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.contrib.auth.models import User
from core.models import Property, PredictionRequest, PredictionResult
import logging

logger = logging.getLogger(__name__)

@api_view(['POST'])
def calculate_prediction(request):
    """API endpoint для расчета прогноза без требования авторизации"""
    try:
        logger.info("Получен запрос на расчет прогноза")
        
        # Получение данных из запроса
        data = request.data
        location_lat = float(data.get('location_lat', 0))
        location_lng = float(data.get('location_lng', 0))
        location_address = data.get('location_address', '')
        area = float(data.get('area', 0))
        property_type = data.get('property_type', 'квартира')
        build_year = int(data.get('build_year', 2023))
        finishing_type = data.get('finishing_type', 'без отделки')
        floor = int(data.get('floor', 1))
        
        logger.info(f"Полученные данные: lat={location_lat}, lng={location_lng}, address={location_address}")

        # Создание объекта недвижимости
        property_obj = Property.objects.create(
            complex_name=location_address,
            region='Москва',
            latitude=location_lat,
            longitude=location_lng,
            area=area,
            property_type=property_type,
            build_year=build_year,
            finishing=finishing_type,
            floor=floor,
            total_floors=floor + 10
        )
        
        logger.info(f"Создан объект недвижимости: {property_obj.id}")

        # Получаем или создаем анонимного пользователя
        anonymous_user, created = User.objects.get_or_create(username='anonymous_user')

        # Создание запроса на прогноз
        prediction_request = PredictionRequest.objects.create(
            user=anonymous_user,  # Используем анонимного пользователя
            property_data=property_obj,
            investment_strategy='перепродажа',
            inflation_rate=4.0,
            central_bank_rate=7.0,
            consumer_price_index=4.0,
            gdp_growth_rate=2.0,
            mortgage_rate=8.0,
            deposit_rate=5.0
        )
        
        logger.info(f"Создан запрос на прогноз: {prediction_request.id}")

        # Генерация прогноза
        from analytics.ml_models import generate_prediction
        prediction_results = generate_prediction(prediction_request)

        if prediction_results is None:
            logger.error("Не удалось получить результаты прогноза")
            return Response({'error': 'Ошибка при генерации прогноза'}, status=500)

        response_data = {
            'success': True,
            'prediction_id': prediction_request.id,
            'results': prediction_results
        }
        
        logger.info("Прогноз успешно сгенерирован")
        return Response(response_data)

    except Exception as e:
        logger.error(f"Ошибка при обработке запроса: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return Response({'error': str(e)}, status=400)

# api/views.py
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.shortcuts import get_object_or_404
from core.models import Property, PredictionRequest, PredictionResult
from .serializers import PropertySerializer, PredictionRequestSerializer, PredictionResultSerializer
import logging

logger = logging.getLogger(__name__)

@api_view(['GET'])
def get_prediction_results(request, prediction_id):
    """API endpoint для получения результатов прогноза"""
    try:
        # Получаем запрос на прогноз
        prediction_request = get_object_or_404(PredictionRequest, id=prediction_id)
        
        # Получаем все результаты для этого запроса
        results = PredictionResult.objects.filter(prediction_request=prediction_request)
        
        # Сериализуем результаты
        serializer = PredictionResultSerializer(results, many=True)
        
        return Response({
            'success': True,
            'results': serializer.data
        })
        
    except PredictionRequest.DoesNotExist:
        return Response({
            'success': False,
            'error': 'Прогноз не найден'
        }, status=404)
    except Exception as e:
        logger.error(f"Ошибка при получении результатов прогноза: {str(e)}")
        return Response({
            'success': False,
            'error': 'Произошла ошибка при получении результатов'
        }, status=500)

@api_view(['GET'])
def get_property_info(request, property_id):
    """API endpoint для получения информации об объекте недвижимости"""
    try:
        # Получаем объект недвижимости
        property_obj = get_object_or_404(Property, id=property_id)
        
        # Сериализуем данные
        serializer = PropertySerializer(property_obj)
        
        return Response({
            'success': True,
            'property': serializer.data
        })
        
    except Property.DoesNotExist:
        return Response({
            'success': False,
            'error': 'Объект не найден'
        }, status=404)
    except Exception as e:
        logger.error(f"Ошибка при получении информации об объекте: {str(e)}")
        return Response({
            'success': False,
            'error': 'Произошла ошибка при получении информации об объекте'
        }, status=500)

@api_view(['GET'])
def check_auth(request):
    """API endpoint для проверки статуса аутентификации"""
    return Response({
        'authenticated': request.user.is_authenticated,
        'username': request.user.username if request.user.is_authenticated else None
    })
