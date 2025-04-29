# core/models.py
# core/models.py
from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone

class District(models.Model):
    """Модель для хранения данных о районах"""
    district_name = models.CharField(max_length=255, verbose_name='Название района')
    district_type = models.CharField(max_length=50, verbose_name='Тип района')
    
    def __str__(self):
        return f"{self.district_name} ({self.district_type})"
    
    class Meta:
        verbose_name = 'Район'
        verbose_name_plural = 'Районы'

class Developer(models.Model):
    """Модель для хранения данных о застройщиках"""
    developer_name = models.CharField(max_length=255, verbose_name='Название застройщика')
    developer_type = models.CharField(max_length=100, verbose_name='Тип застройщика')
    
    def __str__(self):
        return self.developer_name
    
    class Meta:
        verbose_name = 'Застройщик'
        verbose_name_plural = 'Застройщики'

class FinancialInstitution(models.Model):
    """Модель для хранения данных о финансовых учреждениях"""
    institution_name = models.CharField(max_length=255, verbose_name='Название учреждения')
    institution_type = models.CharField(max_length=100, verbose_name='Тип учреждения')
    
    def __str__(self):
        return self.institution_name
    
    class Meta:
        verbose_name = 'Финансовое учреждение'
        verbose_name_plural = 'Финансовые учреждения'

class Property(models.Model):
    """Модель для хранения данных о недвижимости"""
    
    PROPERTY_TYPES = [
        ('квартира', 'Квартира'),
        ('апартаменты', 'Апартаменты'),
        ('коммерческое', 'Коммерческое помещение'),
        ('машиноместо', 'Машиноместо'),
        ('кладовка', 'Кладовка'),
    ]
    
    FINISHING_TYPES = [
        ('без отделки', 'Без отделки'),
        ('черновая', 'Черновая'),
        ('чистовая', 'Чистовая'),
        ('с мебелью', 'С мебелью'),
    ]
    
    ROOM_TYPES = [
        ('студия', 'Студия'),
        ('1-комн', '1-комнатная'),
        ('2-комн', '2-комнатная'),
        ('3-комн', '3-комнатная'),
        ('4+ комн', '4+ комнатная'),
    ]
    
    complex_name = models.CharField(max_length=255, verbose_name='Название ЖК', null=True, blank=True)
    region = models.CharField(max_length=100, verbose_name='Регион', null=True, blank=True)
    floor = models.IntegerField(verbose_name='Этаж', null=True, blank=True)
    property_type = models.CharField(max_length=100, choices=PROPERTY_TYPES, verbose_name='Тип помещения', null=True, blank=True)
    encumbrance_duration = models.IntegerField(verbose_name='Длительность обременения', null=True, blank=True)
    encumbrance_type = models.CharField(max_length=100, verbose_name='Тип обременения', null=True, blank=True)
    assignment = models.BooleanField(verbose_name='Уступка', null=True, blank=True)
    lots_bought = models.IntegerField(verbose_name='Купил лотов в ЖК', null=True, blank=True)
    legal_entity_buyer = models.BooleanField(verbose_name='Покупатель ЮЛ', null=True, blank=True)
    property_class = models.CharField(max_length=50, verbose_name='Класс', null=True, blank=True)
    latitude = models.DecimalField(max_digits=10, decimal_places=8, verbose_name='Широта', null=True, blank=True)
    longitude = models.DecimalField(max_digits=11, decimal_places=8, verbose_name='Долгота', null=True, blank=True)
    mortgage = models.BooleanField(verbose_name='Ипотека', null=True, blank=True)
    finishing = models.CharField(max_length=100, choices=FINISHING_TYPES, verbose_name='Отделка', null=True, blank=True)
    zone = models.CharField(max_length=100, verbose_name='Зона', null=True, blank=True)
    completion_stage = models.CharField(max_length=100, verbose_name='Стадия готовности в дату ДДУ', null=True, blank=True)
    frozen = models.BooleanField(verbose_name='Заморожен', null=True, blank=True)
    pd_issued = models.BooleanField(verbose_name='Выпущена ПД', null=True, blank=True)
    room_type = models.CharField(max_length=50, choices=ROOM_TYPES, verbose_name='Тип комнатности', null=True, blank=True)
    studio = models.BooleanField(verbose_name='Студия', null=True, blank=True)
    price_per_sqm = models.DecimalField(max_digits=12, decimal_places=2, verbose_name='Цена за м²', null=True, blank=True)
    
    # Связи с другими таблицами
    districts = models.ManyToManyField(District, related_name='properties', verbose_name='Районы')
    developers = models.ManyToManyField(Developer, related_name='properties', verbose_name='Застройщики')
    financial_institutions = models.ManyToManyField(FinancialInstitution, related_name='properties', verbose_name='Финансовые учреждения')
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"{self.complex_name or 'Без названия'}, {self.property_type or 'Тип не указан'}, {self.price_per_sqm or 0} руб/м²"
    
    class Meta:
        verbose_name = 'Объект недвижимости'
        verbose_name_plural = 'Объекты недвижимости'

class PropertyDate(models.Model):
    """Модель для хранения дат, связанных с объектами недвижимости"""
    
    DATE_TYPES = [
        ('registration', 'Дата регистрации'),
        ('encumbrance', 'Дата обременения'),
        ('ddu', 'Дата ДДУ'),
        ('sales_start', 'Дата старта продаж'),
        ('completion', 'Дата сдачи')
    ]
    
    property = models.ForeignKey(Property, on_delete=models.CASCADE, related_name='dates')
    date_type = models.CharField(max_length=50, choices=DATE_TYPES, verbose_name='Тип даты')
    date_value = models.DateField(verbose_name='Дата', null=True, blank=True)
    day_of_week = models.IntegerField(verbose_name='День недели', null=True, blank=True)
    month = models.IntegerField(verbose_name='Месяц', null=True, blank=True)
    year = models.IntegerField(verbose_name='Год', null=True, blank=True)
    quarter = models.IntegerField(verbose_name='Квартал', null=True, blank=True)
    
    def __str__(self):
        return f"{self.get_date_type_display()} для {self.property}: {self.date_value or 'Дата не указана'}"
    
    class Meta:
        verbose_name = 'Дата объекта'
        verbose_name_plural = 'Даты объектов'

class PredictionRequest(models.Model):
    """Модель для хранения истории запросов пользователей"""
    
    INVESTMENT_STRATEGIES = [
        ('перепродажа', 'Перепродажа'),
        ('долгосрочная_аренда', 'Долгосрочная аренда'),
        ('краткосрочная_аренда', 'Краткосрочная аренда'),
        ('комбинированная', 'Комбинированная'),
    ]
    
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='prediction_requests')
    property_data = models.ForeignKey(Property, on_delete=models.CASCADE, related_name='predictions')
    
    # Макроэкономические параметры
    inflation_rate = models.FloatField(verbose_name='Инфляция, %', default=4.0)
    central_bank_rate = models.FloatField(verbose_name='Ставка ЦБ, %', default=7.0)
    consumer_price_index = models.FloatField(verbose_name='Индекс потребительских цен, %', default=4.0)
    gdp_growth_rate = models.FloatField(verbose_name='Темп роста ВВП, %', default=2.0)
    mortgage_rate = models.FloatField(verbose_name='Средняя ставка по ипотеке, %', default=8.0)
    deposit_rate = models.FloatField(verbose_name='Средняя доходность депозитов, %', default=5.0)
    
    # Стратегия использования
    investment_strategy = models.CharField(max_length=20, choices=INVESTMENT_STRATEGIES, verbose_name='Стратегия использования')
    
    created_at = models.DateTimeField(default=timezone.now)
    
    def __str__(self):
        return f"Запрос от {self.user.username} ({self.created_at.strftime('%d.%m.%Y %H:%M')})"
    
    class Meta:
        verbose_name = 'Запрос прогноза'
        verbose_name_plural = 'Запросы прогнозов'

class PredictionResult(models.Model):
    """Модель для хранения результатов прогнозов"""
    
    SCENARIO_TYPES = [
        ('positive', 'Позитивный'),
        ('realistic', 'Реалистичный'),
        ('conservative', 'Консервативный'),
    ]
    
    prediction_request = models.ForeignKey(PredictionRequest, on_delete=models.CASCADE, related_name='results')
    scenario_type = models.CharField(max_length=12, choices=SCENARIO_TYPES, verbose_name='Тип сценария')
    
    predicted_price = models.FloatField(verbose_name='Прогнозируемая стоимость')
    price_dynamics_data = models.JSONField(verbose_name='Данные динамики цен')
    comparison_data = models.JSONField(verbose_name='Данные для сравнения')
    
    # Факторы влияния
    location_attractiveness_factor = models.FloatField(verbose_name='Привлекательность района/локации (УРЖ), %')
    transport_accessibility_factor = models.FloatField(verbose_name='Транспортная доступность, %')
    social_infrastructure_factor = models.FloatField(verbose_name='Социальная инфраструктура района, %')
    location_development_factor = models.FloatField(verbose_name='Перспективы развития локации, %')
    macroeconomic_factor = models.FloatField(verbose_name='Макроэкономические факторы, %')
    
    # Инвестиционный сценарий
    annual_yield = models.FloatField(verbose_name='Ожидаемая доходность, % в год')
    investment_horizon = models.IntegerField(verbose_name='Инвестиционный горизонт, лет')
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.get_scenario_type_display()} сценарий для запроса #{self.prediction_request.id}"
    
    class Meta:
        verbose_name = 'Результат прогноза'
        verbose_name_plural = 'Результаты прогнозов'

class ExternalDBConfig(models.Model):
    """Модель для хранения конфигурации подключения к внешней базе данных"""
    
    name = models.CharField(max_length=100, verbose_name='Название')
    host = models.CharField(max_length=255, verbose_name='Хост')
    port = models.IntegerField(verbose_name='Порт')
    database = models.CharField(max_length=100, verbose_name='База данных')
    username = models.CharField(max_length=100, verbose_name='Имя пользователя')
    password = models.CharField(max_length=255, verbose_name='Пароль')
    
    is_active = models.BooleanField(default=True, verbose_name='Активно')
    last_sync = models.DateTimeField(null=True, blank=True, verbose_name='Последняя синхронизация')
    
    def __str__(self):
        return self.name
    
    class Meta:
        verbose_name = 'Конфигурация внешней БД'
        verbose_name_plural = 'Конфигурации внешних БД'
