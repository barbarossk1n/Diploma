# core/views.py
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse, HttpResponse
from django.conf import settings
from .models import Property, PredictionRequest, PredictionResult
import json
import os

def index(request):
    """Представление для главной страницы"""
    context = {
        'yandex_maps_api_key': settings.YANDEX_MAPS_API_KEY,
        'is_authenticated': request.user.is_authenticated,
    }
    return render(request, 'core/index.html', context)

@login_required
def calculate_prediction(request):
    """Обработка запроса на расчет прогноза"""
    if request.method != 'POST':
        return JsonResponse({'error': 'Метод не поддерживается'}, status=405)
    
    # Получение данных из запроса
    try:
        location_lat = float(request.POST.get('location_lat'))
        location_lng = float(request.POST.get('location_lng'))
        location_address = request.POST.get('location_address')
        area = float(request.POST.get('area'))
        property_type = request.POST.get('property_type')
        build_year = int(request.POST.get('build_year'))
        finishing_type = request.POST.get('finishing_type')
        floor = int(request.POST.get('floor'))
        
        # Предполагаем, что общее количество этажей всегда на 10 больше, чем выбранный этаж (можно заменить на фактические данные)
        total_floors = floor + 10 if floor < 40 else 50
        
        # Создание объекта недвижимости
        property_obj = Property.objects.create(
            location_lat=location_lat,
            location_lng=location_lng,
            location_address=location_address,
            area=area,
            property_type=property_type,
            build_year=build_year,
            finishing_type=finishing_type,
            floor=floor,
            total_floors=total_floors
        )
        
        # Создание запроса на прогноз с дефолтными макроэкономическими параметрами
        prediction_request = PredictionRequest.objects.create(
            user=request.user,
            property_data=property_obj,
            investment_strategy='resale'  # По умолчанию - перепродажа
        )
        
        # Здесь будет вызов ML-модели для получения прогноза
        # В реальном проекте это должно быть вынесено в отдельный сервис
        
        # Временные данные для примера (в реальном проекте будут заменены на результаты ML-модели)
        from analytics.ml_models import generate_prediction
        prediction_results = generate_prediction(prediction_request)
        
        return JsonResponse({
            'success': True,
            'prediction_id': prediction_request.id,
            'results': prediction_results
        })
        
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=400)

@login_required
def export_results(request, prediction_id, format='pdf'):
    """Экспорт результатов прогноза"""
    try:
        prediction = PredictionRequest.objects.get(id=prediction_id, user=request.user)
        
        if format == 'pdf':
            # Логика для экспорта в PDF
            from analytics.export import generate_pdf_report
            pdf_buffer = generate_pdf_report(prediction)
            
            # Отправка PDF-файла пользователю
            response = HttpResponse(pdf_buffer.getvalue(), content_type='application/pdf')
            response['Content-Disposition'] = f'attachment; filename="restateval_report_{prediction_id}.pdf"'
            return response
            
        elif format == 'excel':
            # Логика для экспорта в Excel
            from analytics.export import generate_excel_report
            excel_buffer = generate_excel_report(prediction)
            
            # Отправка Excel-файла пользователю
            response = HttpResponse(excel_buffer.getvalue(), content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
            response['Content-Disposition'] = f'attachment; filename="restateval_report_{prediction_id}.xlsx"'
            return response
            
        else:
            return JsonResponse({'error': 'Неподдерживаемый формат'}, status=400)
            
    except PredictionRequest.DoesNotExist:
        return JsonResponse({'error': 'Прогноз не найден'}, status=404)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

