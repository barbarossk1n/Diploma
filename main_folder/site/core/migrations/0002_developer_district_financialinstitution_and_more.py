# Generated by Django 4.2.7 on 2025-04-28 21:45

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ("core", "0001_initial"),
    ]

    operations = [
        migrations.CreateModel(
            name="Developer",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                (
                    "developer_name",
                    models.CharField(
                        max_length=255, verbose_name="Название застройщика"
                    ),
                ),
                (
                    "developer_type",
                    models.CharField(max_length=100, verbose_name="Тип застройщика"),
                ),
            ],
            options={
                "verbose_name": "Застройщик",
                "verbose_name_plural": "Застройщики",
            },
        ),
        migrations.CreateModel(
            name="District",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                (
                    "district_name",
                    models.CharField(max_length=255, verbose_name="Название района"),
                ),
                (
                    "district_type",
                    models.CharField(max_length=50, verbose_name="Тип района"),
                ),
            ],
            options={
                "verbose_name": "Район",
                "verbose_name_plural": "Районы",
            },
        ),
        migrations.CreateModel(
            name="FinancialInstitution",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                (
                    "institution_name",
                    models.CharField(
                        max_length=255, verbose_name="Название учреждения"
                    ),
                ),
                (
                    "institution_type",
                    models.CharField(max_length=100, verbose_name="Тип учреждения"),
                ),
            ],
            options={
                "verbose_name": "Финансовое учреждение",
                "verbose_name_plural": "Финансовые учреждения",
            },
        ),
        migrations.AlterModelOptions(
            name="externaldbconfig",
            options={
                "verbose_name": "Конфигурация внешней БД",
                "verbose_name_plural": "Конфигурации внешних БД",
            },
        ),
        migrations.AlterModelOptions(
            name="predictionrequest",
            options={
                "verbose_name": "Запрос прогноза",
                "verbose_name_plural": "Запросы прогнозов",
            },
        ),
        migrations.AlterModelOptions(
            name="predictionresult",
            options={
                "verbose_name": "Результат прогноза",
                "verbose_name_plural": "Результаты прогнозов",
            },
        ),
        migrations.AlterModelOptions(
            name="property",
            options={
                "verbose_name": "Объект недвижимости",
                "verbose_name_plural": "Объекты недвижимости",
            },
        ),
        migrations.RemoveField(
            model_name="property",
            name="area",
        ),
        migrations.RemoveField(
            model_name="property",
            name="build_year",
        ),
        migrations.RemoveField(
            model_name="property",
            name="finishing_type",
        ),
        migrations.RemoveField(
            model_name="property",
            name="location_address",
        ),
        migrations.RemoveField(
            model_name="property",
            name="location_lat",
        ),
        migrations.RemoveField(
            model_name="property",
            name="location_lng",
        ),
        migrations.RemoveField(
            model_name="property",
            name="total_floors",
        ),
        migrations.AddField(
            model_name="property",
            name="assignment",
            field=models.BooleanField(blank=True, null=True, verbose_name="Уступка"),
        ),
        migrations.AddField(
            model_name="property",
            name="completion_stage",
            field=models.CharField(
                blank=True,
                max_length=100,
                null=True,
                verbose_name="Стадия готовности в дату ДДУ",
            ),
        ),
        migrations.AddField(
            model_name="property",
            name="complex_name",
            field=models.CharField(
                blank=True, max_length=255, null=True, verbose_name="Название ЖК"
            ),
        ),
        migrations.AddField(
            model_name="property",
            name="encumbrance_duration",
            field=models.IntegerField(
                blank=True, null=True, verbose_name="Длительность обременения"
            ),
        ),
        migrations.AddField(
            model_name="property",
            name="encumbrance_type",
            field=models.CharField(
                blank=True, max_length=100, null=True, verbose_name="Тип обременения"
            ),
        ),
        migrations.AddField(
            model_name="property",
            name="finishing",
            field=models.CharField(
                blank=True,
                choices=[
                    ("без отделки", "Без отделки"),
                    ("черновая", "Черновая"),
                    ("чистовая", "Чистовая"),
                    ("с мебелью", "С мебелью"),
                ],
                max_length=100,
                null=True,
                verbose_name="Отделка",
            ),
        ),
        migrations.AddField(
            model_name="property",
            name="frozen",
            field=models.BooleanField(blank=True, null=True, verbose_name="Заморожен"),
        ),
        migrations.AddField(
            model_name="property",
            name="latitude",
            field=models.DecimalField(
                blank=True,
                decimal_places=8,
                max_digits=10,
                null=True,
                verbose_name="Широта",
            ),
        ),
        migrations.AddField(
            model_name="property",
            name="legal_entity_buyer",
            field=models.BooleanField(
                blank=True, null=True, verbose_name="Покупатель ЮЛ"
            ),
        ),
        migrations.AddField(
            model_name="property",
            name="longitude",
            field=models.DecimalField(
                blank=True,
                decimal_places=8,
                max_digits=11,
                null=True,
                verbose_name="Долгота",
            ),
        ),
        migrations.AddField(
            model_name="property",
            name="lots_bought",
            field=models.IntegerField(
                blank=True, null=True, verbose_name="Купил лотов в ЖК"
            ),
        ),
        migrations.AddField(
            model_name="property",
            name="mortgage",
            field=models.BooleanField(blank=True, null=True, verbose_name="Ипотека"),
        ),
        migrations.AddField(
            model_name="property",
            name="pd_issued",
            field=models.BooleanField(
                blank=True, null=True, verbose_name="Выпущена ПД"
            ),
        ),
        migrations.AddField(
            model_name="property",
            name="price_per_sqm",
            field=models.DecimalField(
                blank=True,
                decimal_places=2,
                max_digits=12,
                null=True,
                verbose_name="Цена за м²",
            ),
        ),
        migrations.AddField(
            model_name="property",
            name="property_class",
            field=models.CharField(
                blank=True, max_length=50, null=True, verbose_name="Класс"
            ),
        ),
        migrations.AddField(
            model_name="property",
            name="region",
            field=models.CharField(
                blank=True, max_length=100, null=True, verbose_name="Регион"
            ),
        ),
        migrations.AddField(
            model_name="property",
            name="room_type",
            field=models.CharField(
                blank=True,
                choices=[
                    ("студия", "Студия"),
                    ("1-комн", "1-комнатная"),
                    ("2-комн", "2-комнатная"),
                    ("3-комн", "3-комнатная"),
                    ("4+ комн", "4+ комнатная"),
                ],
                max_length=50,
                null=True,
                verbose_name="Тип комнатности",
            ),
        ),
        migrations.AddField(
            model_name="property",
            name="studio",
            field=models.BooleanField(blank=True, null=True, verbose_name="Студия"),
        ),
        migrations.AddField(
            model_name="property",
            name="zone",
            field=models.CharField(
                blank=True, max_length=100, null=True, verbose_name="Зона"
            ),
        ),
        migrations.AlterField(
            model_name="predictionrequest",
            name="investment_strategy",
            field=models.CharField(
                choices=[
                    ("перепродажа", "Перепродажа"),
                    ("долгосрочная_аренда", "Долгосрочная аренда"),
                    ("краткосрочная_аренда", "Краткосрочная аренда"),
                    ("комбинированная", "Комбинированная"),
                ],
                max_length=20,
                verbose_name="Стратегия использования",
            ),
        ),
        migrations.AlterField(
            model_name="property",
            name="floor",
            field=models.IntegerField(blank=True, null=True, verbose_name="Этаж"),
        ),
        migrations.AlterField(
            model_name="property",
            name="property_type",
            field=models.CharField(
                blank=True,
                choices=[
                    ("квартира", "Квартира"),
                    ("апартаменты", "Апартаменты"),
                    ("коммерческое", "Коммерческое помещение"),
                    ("машиноместо", "Машиноместо"),
                    ("кладовка", "Кладовка"),
                ],
                max_length=100,
                null=True,
                verbose_name="Тип помещения",
            ),
        ),
        migrations.CreateModel(
            name="PropertyDate",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                (
                    "date_type",
                    models.CharField(
                        choices=[
                            ("registration", "Дата регистрации"),
                            ("encumbrance", "Дата обременения"),
                            ("ddu", "Дата ДДУ"),
                            ("sales_start", "Дата старта продаж"),
                            ("completion", "Дата сдачи"),
                        ],
                        max_length=50,
                        verbose_name="Тип даты",
                    ),
                ),
                (
                    "date_value",
                    models.DateField(blank=True, null=True, verbose_name="Дата"),
                ),
                (
                    "day_of_week",
                    models.IntegerField(
                        blank=True, null=True, verbose_name="День недели"
                    ),
                ),
                (
                    "month",
                    models.IntegerField(blank=True, null=True, verbose_name="Месяц"),
                ),
                (
                    "year",
                    models.IntegerField(blank=True, null=True, verbose_name="Год"),
                ),
                (
                    "quarter",
                    models.IntegerField(blank=True, null=True, verbose_name="Квартал"),
                ),
                (
                    "property",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="dates",
                        to="core.property",
                    ),
                ),
            ],
            options={
                "verbose_name": "Дата объекта",
                "verbose_name_plural": "Даты объектов",
            },
        ),
        migrations.AddField(
            model_name="property",
            name="developers",
            field=models.ManyToManyField(
                related_name="properties",
                to="core.developer",
                verbose_name="Застройщики",
            ),
        ),
        migrations.AddField(
            model_name="property",
            name="districts",
            field=models.ManyToManyField(
                related_name="properties", to="core.district", verbose_name="Районы"
            ),
        ),
        migrations.AddField(
            model_name="property",
            name="financial_institutions",
            field=models.ManyToManyField(
                related_name="properties",
                to="core.financialinstitution",
                verbose_name="Финансовые учреждения",
            ),
        ),
    ]
