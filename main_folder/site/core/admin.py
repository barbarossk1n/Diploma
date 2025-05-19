# core/admin.py

from django.contrib import admin
from .models import (
    District, Developer, FinancialInstitution, 
    Property, PredictionRequest, PredictionResult,
    PropertyDistrict, PropertyDeveloper, PropertyInstitution
)

@admin.register(District)
class DistrictAdmin(admin.ModelAdmin):
    list_display = ['district_name', 'district_type']
    list_filter = ['district_type']
    search_fields = ['district_name']
    ordering = ['district_name']

@admin.register(Developer)
class DeveloperAdmin(admin.ModelAdmin):
    list_display = ['developer_name', 'developer_type']
    list_filter = ['developer_type']
    search_fields = ['developer_name']
    ordering = ['developer_name']

@admin.register(FinancialInstitution)
class FinancialInstitutionAdmin(admin.ModelAdmin):
    list_display = ['institution_name', 'institution_type']
    list_filter = ['institution_type']
    search_fields = ['institution_name']
    ordering = ['institution_name']

@admin.register(Property)
class PropertyAdmin(admin.ModelAdmin):
    list_display = [
        'location_address',
        'property_type',
        'property_class',
        'floor',
        'finishing',
    ]
    list_filter = [
        'property_type',
        'property_class',
        'finishing',
    ]
    search_fields = [
        'location_address',
    ]
    
    # Удаляем filter_horizontal, так как используем through models
    
    fieldsets = (
        ('Основная информация', {
            'fields': (
                'location_address',
                'latitude',
                'longitude',
                'floor',
            )
        }),
        ('Характеристики', {
            'fields': (
                'complex_name',
                'property_type',
                'property_class',
                'finishing',
                'room_type',
                'studio',
                'price_per_sqm',
            )
        }),
        ('Дополнительные характеристики', {
            'fields': (
                'encumbrance_duration',
                'encumbrance_type',
                'assignment',
                'lots_bought',
                'legal_entity_buyer',
                'mortgage',
                'zone',
                'completion_stage',
                'frozen',
                'pd_issued',
            )
        }),
    )

# Добавьте инлайн-модели для связей
class PropertyDistrictInline(admin.TabularInline):
    model = PropertyDistrict
    extra = 1

class PropertyDeveloperInline(admin.TabularInline):
    model = PropertyDeveloper
    extra = 1

class PropertyInstitutionInline(admin.TabularInline):
    model = PropertyInstitution
    extra = 1

# Добавьте инлайны в PropertyAdmin
PropertyAdmin.inlines = [
    PropertyDistrictInline,
    PropertyDeveloperInline,
    PropertyInstitutionInline,
]


@admin.register(PredictionRequest)
class PredictionRequestAdmin(admin.ModelAdmin):
    list_display = [
        'id',
        'user',
        'property_data',
        'investment_strategy',
        'created_at'
    ]
    list_filter = [
        'investment_strategy',
        'created_at',
        'user',
    ]
    search_fields = [
        'user__username',
        'property_data__location_address'
    ]
    readonly_fields = ['created_at']
    fieldsets = (
        ('Основная информация', {
            'fields': (
                'user',
                'property_data',
                'investment_strategy',
            )
        }),
        ('Макроэкономические параметры', {
            'fields': (
                'inflation_rate',
                'central_bank_rate',
                'consumer_price_index',
                'gdp_growth_rate',
                'mortgage_rate',
                'deposit_rate',
            )
        }),
        ('Метаданные', {
            'fields': ('created_at',)
        }),
    )

@admin.register(PredictionResult)
class PredictionResultAdmin(admin.ModelAdmin):
    list_display = [
        'id', 
        'prediction_request', 
        'scenario_type', 
        'predicted_price',
        'created_at'
    ]
    list_filter = [
        'scenario_type',
        'created_at'
    ]
    search_fields = [
        'prediction_request__id',
        'prediction_request__property_data__location_address'
    ]
    readonly_fields = ['created_at']
    fieldsets = (
        ('Основная информация', {
            'fields': (
                'prediction_request',
                'scenario_type',
                'predicted_price',
            )
        }),
        ('Дополнительные данные', {
            'fields': (
                'influence_factors',
                'price_dynamics',
            )
        }),
        ('Метаданные', {
            'fields': ('created_at',)
        }),
    )



