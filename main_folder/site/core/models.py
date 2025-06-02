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
        db_table = 'districts'

class Developer(models.Model):
    """Модель для хранения данных о застройщиках"""
    developer_name = models.CharField(max_length=255, verbose_name='Название застройщика')
    developer_type = models.CharField(max_length=100, verbose_name='Тип застройщика')
    
    def __str__(self):
        return self.developer_name
    
    class Meta:
        verbose_name = 'Застройщик'
        verbose_name_plural = 'Застройщики'
        db_table = 'developers'

class FinancialInstitution(models.Model):
    """Модель для хранения данных о финансовых учреждениях"""
    institution_name = models.CharField(max_length=255, verbose_name='Название учреждения')
    institution_type = models.CharField(max_length=100, verbose_name='Тип учреждения')
    
    def __str__(self):
        return self.institution_name
    
    class Meta:
        verbose_name = 'Финансовое учреждение'
        verbose_name_plural = 'Финансовые учреждения'
        db_table = 'financial_institutions'
        
class Property(models.Model):
    complex_name = models.CharField(max_length=255, null=True, blank=True)
    region = models.CharField(max_length=100, null=True, blank=True)
    location_address = models.CharField(max_length=255, null=True, blank=True)
    floor = models.IntegerField(null=True, blank=True)
    property_type = models.CharField(max_length=100, null=True, blank=True)
    property_class = models.CharField(max_length=50, null=True, blank=True)
    latitude = models.DecimalField(max_digits=10, decimal_places=8, null=True, blank=True)
    longitude = models.DecimalField(max_digits=11, decimal_places=8, null=True, blank=True)
    finishing = models.CharField(max_length=100, null=True, blank=True)
    purchase_date = models.DateField(null=True, blank=True)

    # Связи
    districts = models.ManyToManyField('District', through='PropertyDistrict')
    developers = models.ManyToManyField('Developer', through='PropertyDeveloper')
    financial_institutions = models.ManyToManyField('FinancialInstitution', through='PropertyInstitution')

    class Meta:
        db_table = 'properties'




class PropertyDistrict(models.Model):
    property = models.ForeignKey(Property, on_delete=models.CASCADE)
    district = models.ForeignKey(District, on_delete=models.CASCADE)

    class Meta:
        db_table = 'properties_districts'
        unique_together = ('property', 'district')

class PropertyDeveloper(models.Model):
    property = models.ForeignKey(Property, on_delete=models.CASCADE)
    developer = models.ForeignKey(Developer, on_delete=models.CASCADE)

    class Meta:
        db_table = 'properties_developers'
        unique_together = ('property', 'developer')

class PropertyInstitution(models.Model):
    property = models.ForeignKey(Property, on_delete=models.CASCADE)
    institution = models.ForeignKey(FinancialInstitution, on_delete=models.CASCADE)

    class Meta:
        db_table = 'properties_institutions'
        unique_together = ('property', 'institution')

class Date(models.Model):
    property = models.ForeignKey(Property, on_delete=models.CASCADE)
    date_type = models.CharField(max_length=50)
    date_value = models.DateField(null=True, blank=True)
    day_of_week = models.IntegerField(null=True, blank=True)
    month = models.IntegerField(null=True, blank=True)
    year = models.IntegerField(null=True, blank=True)
    quarter = models.IntegerField(null=True, blank=True)

    class Meta:
        db_table = 'dates'

class PredictionRequest(models.Model):
    """Модель для хранения запросов на прогноз"""
    property_data = models.ForeignKey(Property, on_delete=models.CASCADE)
    created_at = models.DateTimeField(default=timezone.now)
    status = models.CharField(max_length=50, default='pending')
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    investment_strategy = models.CharField(max_length=100, null=True, blank=True)
    inflation_rate = models.DecimalField(max_digits=5, decimal_places=2, null=True, blank=True)
    central_bank_rate = models.DecimalField(max_digits=5, decimal_places=2, null=True, blank=True)
    consumer_price_index = models.DecimalField(max_digits=5, decimal_places=2, null=True, blank=True)
    gdp_growth_rate = models.DecimalField(max_digits=5, decimal_places=2, null=True, blank=True)
    mortgage_rate = models.DecimalField(max_digits=5, decimal_places=2, null=True, blank=True)
    deposit_rate = models.DecimalField(max_digits=5, decimal_places=2, null=True, blank=True)

    def __str__(self):
        return f"Запрос #{self.id} ({self.created_at.strftime('%d.%m.%Y %H:%M')})"

    class Meta:
        verbose_name = 'Запрос прогноза'
        verbose_name_plural = 'Запросы прогнозов'
        db_table = 'prediction_requests'

class PredictionResult(models.Model):
    """Модель для хранения результатов прогноза"""
    prediction_request = models.ForeignKey(PredictionRequest, on_delete=models.CASCADE, related_name='results')
    scenario_type = models.CharField(max_length=50)
    predicted_price = models.DecimalField(max_digits=12, decimal_places=2)
    influence_factors = models.JSONField()
    price_dynamics = models.JSONField()
    created_at = models.DateTimeField(default=timezone.now)

    def __str__(self):
        return f"Результат для запроса #{self.prediction_request.id} ({self.scenario_type})"

    class Meta:
        verbose_name = 'Результат прогноза'
        verbose_name_plural = 'Результаты прогнозов'
        db_table = 'prediction_results'
        ordering = ['prediction_request', 'scenario_type']

        verbose_name_plural = 'Конфигурации внешних БД'


