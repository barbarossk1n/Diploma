# core/admin.py
from django.contrib import admin
from .models import (
    Property, PredictionRequest, PredictionResult, 
    District, Developer, FinancialInstitution, PropertyDate, 
    ExternalDBConfig
)

@admin.register(District)
class DistrictAdmin(admin.ModelAdmin):
    list_display = ('district_name', 'district_type')
    list_filter = ('district_type',)
    search_fields = ('district_name',)

@admin.register(Developer)
class DeveloperAdmin(admin.ModelAdmin):
    list_display = ('developer_name', 'developer_type')
    list_filter = ('developer_type',)
    search_fields = ('developer_name',)

@admin.register(FinancialInstitution)
class FinancialInstitutionAdmin(admin.ModelAdmin):
    list_display = ('institution_name', 'institution_type')
    list_filter = ('institution_type',)
    search_fields = ('institution_name',)

@admin.register(Property)
class PropertyAdmin(admin.ModelAdmin):
    list_display = ('complex_name', 'property_type', 'room_type', 'price_per_sqm', 'region')
    list_filter = ('property_type', 'room_type', 'region', 'property_class')
    search_fields = ('complex_name', 'region')
    filter_horizontal = ('districts', 'developers', 'financial_institutions')
    fieldsets = (
        ('Основная информация', {
            'fields': ('complex_name', 'region', 'property_type', 'room_type', 'studio', 'price_per_sqm')
        }),
        ('Расположение', {
            'fields': ('latitude', 'longitude', 'districts')
        }),
        ('Характеристики объекта', {
            'fields': ('floor', 'property_class', 'finishing', 'zone', 'completion_stage')
        }),
        ('Статусы', {
            'fields': ('mortgage', 'assignment', 'legal_entity_buyer', 'frozen', 'pd_issued')
        }),
        ('Связи', {
            'fields': ('developers', 'financial_institutions')
        }),
    )

@admin.register(PropertyDate)
class PropertyDateAdmin(admin.ModelAdmin):
    list_display = ('property', 'date_type', 'date_value', 'year', 'month', 'quarter')
    list_filter = ('date_type', 'year', 'quarter')
    search_fields = ('property__complex_name',)

@admin.register(PredictionRequest)
class PredictionRequestAdmin(admin.ModelAdmin):
    list_display = ('user', 'property_data', 'investment_strategy', 'created_at')
    list_filter = ('investment_strategy', 'created_at')
    fieldsets = (
        ('Основная информация', {
            'fields': ('user', 'property_data', 'investment_strategy')
        }),
        ('Макроэкономические параметры', {
            'fields': ('inflation_rate', 'central_bank_rate', 'consumer_price_index', 
                      'gdp_growth_rate', 'mortgage_rate', 'deposit_rate')
        }),
    )

@admin.register(PredictionResult)
class PredictionResultAdmin(admin.ModelAdmin):
    list_display = ('prediction_request', 'scenario_type', 'predicted_price', 'annual_yield')
    list_filter = ('scenario_type',)
    fieldsets = (
        ('Основная информация', {
            'fields': ('prediction_request', 'scenario_type', 'predicted_price', 'annual_yield', 'investment_horizon')
        }),
        ('Факторы влияния', {
            'fields': ('location_attractiveness_factor', 'transport_accessibility_factor', 
                      'social_infrastructure_factor', 'location_development_factor', 'macroeconomic_factor')
        }),
        ('Данные для графиков', {
            'fields': ('price_dynamics_data', 'comparison_data')
        }),
    )

@admin.register(ExternalDBConfig)
class ExternalDBConfigAdmin(admin.ModelAdmin):
    list_display = ('name', 'host', 'database', 'is_active', 'last_sync')
    list_filter = ('is_active',)
    search_fields = ('name', 'host')
    fieldsets = (
        ('Основная информация', {
            'fields': ('name', 'is_active')
        }),
        ('Параметры подключения', {
            'fields': ('host', 'port', 'database', 'username', 'password')
        }),
        ('Синхронизация', {
            'fields': ('last_sync',)
        }),
    )

