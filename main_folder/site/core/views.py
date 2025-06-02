# core/views.py
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse, HttpResponse
from django.conf import settings
from django.core.cache import cache
from django.views.decorators.cache import cache_page
from django.views.decorators.csrf import ensure_csrf_cookie
from django.core.exceptions import ValidationError
from .models import Property, PredictionRequest, PredictionResult
from .serializers import PredictionResultSerializer
from analytics.ml_models import generate_prediction

from datetime import datetime
import logging
import json

# Настройка логирования
logger = logging.getLogger(__name__)

# Маппинг классов недвижимости
class_mapping = {
    1: "эконом",
    2: "комфорт",
    3: "комфорт+",
    4: "бизнес",
    5: "элит"
}

@ensure_csrf_cookie
def index(request):
    """Главная страница"""
    try:
        context = {
            'yandex_maps_api_key': settings.YANDEX_MAPS_API_KEY,
            'is_authenticated': request.user.is_authenticated,
        }
        return render(request, 'core/index.html', context)
    except Exception as e:
        logger.error(f"Error rendering index page: {str(e)}")
        return render(request, 'core/500.html', status=500)

def validate_property_data(data):
    """
    Валидация входных данных о недвижимости
    
    Args:
        data (dict): Словарь с данными о недвижимости
        
    Raises:
        ValueError: Если данные некорректны
    """
    required_fields = [
        'area', 'floor', 'property_type', 'property_class',
        'location_lat', 'location_lng', 'location_address'
    ]
    
    # Проверка наличия обязательных полей
    missing_fields = [field for field in required_fields if not data.get(field)]
    if missing_fields:
        raise ValueError(f"Отсутствуют обязательные поля: {', '.join(missing_fields)}")
    
    # Валидация типов данных
    try:
        float(data.get('area'))
        int(data.get('floor'))
        float(data.get('location_lat'))
        float(data.get('location_lng'))
        int(data.get('property_class'))
        int(data.get('purchase_month'))
        int(data.get('purchase_year'))
    except (ValueError, TypeError) as e:
        raise ValueError(f"Некорректный формат числовых данных: {str(e)}")

@cache_page(60 * 15)
def calculate_prediction(request):
    """
    Обработка запроса на расчет прогноза
    
    Returns:
        JsonResponse: Результаты прогноза или сообщение об ошибке
    """
    if request.method != 'POST':
        return JsonResponse({'error': 'Метод не поддерживается'}, status=405)
    
    try:
        # Валидация входных данных
        try:
            validate_property_data(request.POST)
        except ValueError as e:
            return JsonResponse({'error': str(e)}, status=400)
            
        # Подготовка данных о недвижимости
        property_data = {
            'complex_name': request.POST.get('location_address'),
            'location_address': request.POST.get('location_address'),
            'region': 'Москва',
            'latitude': float(request.POST.get('location_lat')),
            'longitude': float(request.POST.get('location_lng')),
            'floor': int(request.POST.get('floor')),
            'property_type': request.POST.get('property_type'),
            'property_class': class_mapping.get(
                int(request.POST.get('property_class')), 
                'эконом'
            ),
            'finishing': request.POST.get('finishing_type'),
            'purchase_date': datetime(
                int(request.POST.get('purchase_year')),
                int(request.POST.get('purchase_month')),
                1
            ).date()
        }

        # Создание объекта недвижимости
        try:
            property_obj = Property.objects.create(**property_data)
        except Exception as e:
            logger.error(f"Error creating property object: {str(e)}")
            return JsonResponse({
                'error': 'Ошибка при сохранении данных объекта'
            }, status=500)

        # Создание запроса на прогноз и генерация результатов
        try:
            prediction_request = PredictionRequest.objects.create(
                user=request.user if request.user.is_authenticated else None,
                property_data=property_obj
            )
            
            results = generate_prediction(prediction_request)
            if not results:
                raise ValueError("Не удалось сгенерировать прогноз")

            serializer = PredictionResultSerializer(results, many=True)
            
            return JsonResponse({
                'success': True,
                'prediction_id': prediction_request.id,
                'results': serializer.data
            })
            
        except Exception as e:
            logger.error(f"Error generating prediction: {str(e)}")
            property_obj.delete()
            return JsonResponse({'error': str(e)}, status=500)

    except Exception as e:
        logger.error(f"Unhandled error in calculate_prediction: {str(e)}")
        return JsonResponse({
            'error': 'Внутренняя ошибка сервера'
        }, status=500)

@login_required
def export_results(request, prediction_id, format='pdf'):
    """
    Экспорт результатов прогноза
    
    Args:
        prediction_id (int): ID прогноза
        format (str): Формат экспорта (pdf/excel)
        
    Returns:
        HttpResponse: Файл отчета или сообщение об ошибке
    """
    try:
        prediction = PredictionRequest.objects.get(
            id=prediction_id,
            user=request.user
        )
        
        if format == 'pdf':
            from analytics.export import generate_pdf_report
            pdf_buffer = generate_pdf_report(prediction)
            response = HttpResponse(
                pdf_buffer.getvalue(), 
                content_type='application/pdf'
            )
            response['Content-Disposition'] = (
                f'attachment; filename="restateval_report_{prediction_id}.pdf"'
            )
            return response
            
        elif format == 'excel':
            from analytics.export import generate_excel_report
            excel_buffer = generate_excel_report(prediction)
            response = HttpResponse(
                excel_buffer.getvalue(),
                content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )
            response['Content-Disposition'] = (
                f'attachment; filename="restateval_report_{prediction_id}.xlsx"'
            )
            return response
            
        else:
            return JsonResponse(
                {'error': 'Неподдерживаемый формат экспорта'}, 
                status=400
            )
            
    except PredictionRequest.DoesNotExist:
        return JsonResponse({'error': 'Прогноз не найден'}, status=404)
        
    except Exception as e:
        logger.error(f"Error exporting results: {str(e)}")
        return JsonResponse(
            {'error': 'Ошибка при формировании отчета'}, 
            status=500
        )

@login_required
def get_prediction_history(request):
    """
    Получение истории прогнозов пользователя
    
    Returns:
        JsonResponse: История прогнозов или сообщение об ошибке
    """
    try:
        predictions = PredictionRequest.objects.filter(
            user=request.user
        ).order_by('-created_at')[:10]
        
        history = []
        for pred in predictions:
            realistic_result = pred.results.filter(
                scenario_type='realistic'
            ).first()
            
            if realistic_result:
                history.append({
                    'id': pred.id,
                    'date': pred.created_at.strftime('%d.%m.%Y %H:%M'),
                    'address': pred.property_data.location_address,
                    'predicted_price': realistic_result.predicted_price,
                    'property_type': pred.property_data.property_type,
                    'floor': pred.property_data.floor
                })
        
        return JsonResponse({'success': True, 'history': history})
        
    except Exception as e:
        logger.error(f"Error getting prediction history: {str(e)}")
        return JsonResponse(
            {'error': 'Ошибка при получении истории'}, 
            status=500
        )

@login_required
def delete_prediction(request, prediction_id):
    """
    Удаление прогноза
    
    Args:
        prediction_id (int): ID прогноза
        
    Returns:
        JsonResponse: Статус операции
    """
    try:
        prediction = PredictionRequest.objects.get(
            id=prediction_id,
            user=request.user
        )
        
        property_obj = prediction.property_data
        prediction.delete()
        property_obj.delete()
        
        return JsonResponse({'success': True})
        
    except PredictionRequest.DoesNotExist:
        return JsonResponse({'error': 'Прогноз не найден'}, status=404)
        
    except Exception as e:
        logger.error(f"Error deleting prediction: {str(e)}")
        return JsonResponse({'error': 'Ошибка при удалении'}, status=500)

def handler404(request, exception):
    """Обработчик ошибки 404"""
    return render(request, 'core/404.html', status=404)

def handler500(request):
    """Обработчик ошибки 500"""
    return render(request, 'core/500.html', status=500)



