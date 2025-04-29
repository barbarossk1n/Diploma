# api/views.py
from rest_framework import viewsets
from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from core.models import Property, PredictionRequest, PredictionResult
from .serializers import PropertySerializer, PredictionRequestSerializer, PredictionResultSerializer

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def user_predictions(request):
    """Получение списка прогнозов пользователя"""
    predictions = PredictionRequest.objects.filter(user=request.user).order_by('-created_at')
    serializer = PredictionRequestSerializer(predictions, many=True)
    return Response(serializer.data)

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def prediction_detail(request, prediction_id):
    """Получение детальной информации о прогнозе"""
    try:
        prediction = PredictionRequest.objects.get(id=prediction_id, user=request.user)
        prediction_serializer = PredictionRequestSerializer(prediction)
        
        results = PredictionResult.objects.filter(prediction_request=prediction)
        results_serializer = PredictionResultSerializer(results, many=True)
        
        return Response({
            'prediction': prediction_serializer.data,
            'results': results_serializer.data
        })
    except PredictionRequest.DoesNotExist:
        return Response({'error': 'Прогноз не найден'}, status=404)
