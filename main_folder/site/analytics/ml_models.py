# analytics/ml_models.py
# analytics/ml_models.py
import os
import json
import pickle
import numpy as np
from datetime import datetime, timedelta
from django.conf import settings

# Путь к предобученной модели
MODEL_PATH = os.path.join(settings.BASE_DIR, 'analytics/models/xgboost_model.pkl')

def load_model():
    """Загрузка предобученной модели"""
    try:
        # Проверяем, существует ли файл модели
        if not os.path.exists(MODEL_PATH):
            print(f"Файл модели не найден: {MODEL_PATH}")
            return None
            
        with open(MODEL_PATH, 'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        print(f"Ошибка загрузки модели: {e}")
        # Возвращаем None в случае ошибки
        return None

def prepare_features(prediction_request):
    """Подготовка признаков для модели на основе запроса"""
    property_data = prediction_request.property_data
    
    # Преобразование данных в формат, подходящий для модели
    features = {
        'complex_name': property_data.complex_name,
        'region': property_data.region,
        'floor': property_data.floor,
        'property_type': property_data.property_type,
        'latitude': property_data.latitude,
        'longitude': property_data.longitude,
        'finishing': property_data.finishing,
        'room_type': property_data.room_type,
        'studio': property_data.studio,
        'year': property_data.build_year,  # Соответствует году постройки
        'inflation_rate': prediction_request.inflation_rate,
        'central_bank_rate': prediction_request.central_bank_rate,
        'consumer_price_index': prediction_request.consumer_price_index,
        'gdp_growth_rate': prediction_request.gdp_growth_rate,
        'mortgage_rate': prediction_request.mortgage_rate,
        'deposit_rate': prediction_request.deposit_rate,
        'investment_strategy': prediction_request.investment_strategy
    }
    
    # Преобразование категориальных признаков (one-hot encoding)
    # Типы помещений
    property_types = ['квартира', 'апартаменты', 'коммерческое', 'машиноместо', 'кладовка']
    for pt in property_types:
        features[f'property_type_{pt}'] = 1 if property_data.property_type == pt else 0
    
    # Типы отделки
    finishing_types = ['без отделки', 'черновая', 'чистовая', 'с мебелью']
    for ft in finishing_types:
        features[f'finishing_{ft}'] = 1 if property_data.finishing == ft else 0
    
    # Типы комнатности
    room_types = ['студия', '1-комн', '2-комн', '3-комн', '4+ комн']
    for rt in room_types:
        features[f'room_type_{rt}'] = 1 if property_data.room_type == rt else 0
    
    # Инвестиционные стратегии
    investment_strategies = ['перепродажа', 'долгосрочная_аренда', 'краткосрочная_аренда', 'комбинированная']
    for ist in investment_strategies:
        features[f'investment_strategy_{ist}'] = 1 if prediction_request.investment_strategy == ist else 0
    
    # Преобразование в numpy массив для модели
    feature_names = [
        'latitude', 'longitude', 'floor', 'year', 
        'inflation_rate', 'central_bank_rate', 'consumer_price_index', 
        'gdp_growth_rate', 'mortgage_rate', 'deposit_rate',
        'property_type_квартира', 'property_type_апартаменты', 'property_type_коммерческое', 
        'property_type_машиноместо', 'property_type_кладовка',
        'finishing_без отделки', 'finishing_черновая', 'finishing_чистовая', 'finishing_с мебелью',
        'room_type_студия', 'room_type_1-комн', 'room_type_2-комн', 'room_type_3-комн', 'room_type_4+ комн',
        'investment_strategy_перепродажа', 'investment_strategy_долгосрочная_аренда', 
        'investment_strategy_краткосрочная_аренда', 'investment_strategy_комбинированная',
        'studio'
    ]
    
    features_array = np.array([[features.get(name, 0) for name in feature_names]])
    return features_array

def generate_price_dynamics(base_price, years=3, months_per_year=12, scenario_type='realistic'):
    """Генерация данных динамики цен для графиков"""
    total_months = years * months_per_year
    start_date = datetime.now()
    
    # Коэффициенты роста для разных сценариев
    growth_factors = {
        'positive': 0.008,  # ~10% годовых
        'realistic': 0.005,  # ~6% годовых
        'conservative': 0.002  # ~2.5% годовых
    }
    
    # Волатильность для разных сценариев
    volatility = {
        'positive': 0.002,
        'realistic': 0.004,
        'conservative': 0.006
    }
    
    # Генерация временного ряда цен
    prices = []
    current_price = base_price
    
    for i in range(total_months):
        # Добавляем случайную волатильность к базовому тренду
        random_factor = np.random.normal(0, volatility[scenario_type])
        growth = growth_factors[scenario_type] + random_factor
        current_price *= (1 + growth)
        
        # Добавляем сезонность (летом цены немного выше)
        month = (start_date.month + i) % 12
        if month in [5, 6, 7]:  # Лето
            current_price *= 1.005
        elif month in [11, 0, 1]:  # Зима
            current_price *= 0.995
            
        prices.append(round(current_price, 2))
    
    # Формирование дат для оси X
    dates = [(start_date + timedelta(days=i*30)).strftime('%Y-%m-%d') for i in range(total_months)]
    
    return {
        'dates': dates,
        'prices': prices
    }

def generate_comparison_data(base_price, scenario_type='realistic'):
    """Генерация данных для сравнительного графика"""
    years = list(range(1, 6))  # 5 лет
    
    # Доходность для разных типов инвестиций
    yields = {
        'property': {
            'positive': [12, 11, 10.5, 10, 9.5],
            'realistic': [8, 7.5, 7, 6.5, 6],
            'conservative': [4, 3.8, 3.6, 3.4, 3.2]
        },
        'stocks': {
            'positive': [15, 14, 13, 12, 11],
            'realistic': [10, 9.5, 9, 8.5, 8],
            'conservative': [5, 4.5, 4, 3.5, 3]
        },
        'bonds': {
            'positive': [8, 7.8, 7.6, 7.4, 7.2],
            'realistic': [6, 5.8, 5.6, 5.4, 5.2],
            'conservative': [3, 2.9, 2.8, 2.7, 2.6]
        },
        'deposits': {
            'positive': [7, 6.8, 6.6, 6.4, 6.2],
            'realistic': [5, 4.9, 4.8, 4.7, 4.6],
            'conservative': [2.5, 2.4, 2.3, 2.2, 2.1]
        }
    }
    
    # Расчет стоимости для каждого типа инвестиций
    result = {
        'years': years,
        'property': [],
        'stocks': [],
        'bonds': [],
        'deposits': []
    }
    
    for investment_type in ['property', 'stocks', 'bonds', 'deposits']:
        current_value = base_price
        values = [current_value]
        
        for i in range(len(years) - 1):
            annual_yield = yields[investment_type][scenario_type][i] / 100
            current_value *= (1 + annual_yield)
            values.append(round(current_value, 2))
            
        result[investment_type] = values
    
    return result

def calculate_influence_factors(prediction_request, scenario_type='realistic'):
    """Расчет факторов влияния на прогноз"""
    # В реальном проекте эти значения должны рассчитываться на основе модели
    # Здесь приведены примерные значения для демонстрации
    
    if scenario_type == 'positive':
        return {
            'location_attractiveness': 35.0,
            'transport_accessibility': 25.0,
            'social_infrastructure': 15.0,
            'location_development': 15.0,
            'macroeconomic': 10.0
        }
    elif scenario_type == 'realistic':
        return {
            'location_attractiveness': 30.0,
            'transport_accessibility': 20.0,
            'social_infrastructure': 20.0,
            'location_development': 15.0,
            'macroeconomic': 15.0
        }
    else:  # conservative
        return {
            'location_attractiveness': 25.0,
            'transport_accessibility': 20.0,
            'social_infrastructure': 15.0,
            'location_development': 15.0,
            'macroeconomic': 25.0
        }

def calculate_annual_yield(scenario_type='realistic'):
    """Расчет ожидаемой годовой доходности"""
    if scenario_type == 'positive':
        return 10.5
    elif scenario_type == 'realistic':
        return 7.0
    else:  # conservative
        return 3.5

def get_district_factor(property_data):
    """Получение фактора влияния района на стоимость"""
    # В реальном проекте следует запросить эти данные из БД
    # Здесь просто возвращаем значение по умолчанию
    return 1.0

def get_developer_factor(property_data):
    """Получение фактора влияния застройщика на стоимость"""
    # В реальном проекте следует запросить эти данные из БД
    # Здесь просто возвращаем значение по умолчанию
    return 1.0

def generate_prediction(prediction_request):
    """Генерация прогноза на основе ML-модели"""
    # Загрузка модели
    model = load_model()
    
    if model is None:
        # Если модель не загружена, используем заглушку
        # Используем price_per_sqm как основу для расчета
        base_price = prediction_request.property_data.area * 150000  # Примерная цена: 150 000 руб/м²
    else:
        # Подготовка признаков для модели
        features = prepare_features(prediction_request)
        
        # Получение прогноза от модели - цена за квадратный метр
        price_per_sqm = model.predict(features)[0]
        
        # Расчет общей стоимости
        base_price = price_per_sqm * prediction_request.property_data.area
    
    # Получение дополнительных факторов из связанных таблиц
    district_factor = get_district_factor(prediction_request.property_data)
    developer_factor = get_developer_factor(prediction_request.property_data)
    
    # Корректировка цены с учетом факторов
    base_price = base_price * district_factor * developer_factor
    
    # Генерация результатов для разных сценариев
    scenarios = ['positive', 'realistic', 'conservative']
    results = {}
    
    for scenario in scenarios:
        # Корректировка базовой цены в зависимости от сценария
        if scenario == 'positive':
            price = base_price * 1.1
        elif scenario == 'realistic':
            price = base_price
        else:  # conservative
            price = base_price * 0.9
            
        # Расчет факторов влияния
        influence_factors = calculate_influence_factors(prediction_request, scenario)
        
        # Расчет ожидаемой доходности
        annual_yield = calculate_annual_yield(scenario)
        
        # Генерация данных для графиков
        price_dynamics = generate_price_dynamics(price, scenario_type=scenario)
        comparison_data = generate_comparison_data(price, scenario_type=scenario)
        
        # Сохранение результата в базе данных
        from core.models import PredictionResult
        prediction_result = PredictionResult.objects.create(
            prediction_request=prediction_request,
            scenario_type=scenario,
            predicted_price=price,
            price_dynamics_data=price_dynamics,
            comparison_data=comparison_data,
            location_attractiveness_factor=influence_factors['location_attractiveness'],
            transport_accessibility_factor=influence_factors['transport_accessibility'],
            social_infrastructure_factor=influence_factors['social_infrastructure'],
            location_development_factor=influence_factors['location_development'],
            macroeconomic_factor=influence_factors['macroeconomic'],
            annual_yield=annual_yield,
            investment_horizon=5  # Предполагаем горизонт инвестирования 5 лет
        )
        
        # Добавление результатов в ответ
        results[scenario] = {
            'predicted_price': price,
            'price_dynamics': price_dynamics,
            'comparison_data': comparison_data,
            'influence_factors': influence_factors,
            'annual_yield': annual_yield,
            'investment_horizon': 5
        }
    
    return results
