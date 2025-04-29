# real_estate_db_connector.py
import psycopg2
import pandas as pd
import numpy as np
from psycopg2.extras import RealDictCursor
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import sys

class RealEstateDB:
    """Класс для работы с базой данных недвижимости"""
    
    def __init__(self, db_config=None):
        """
        Инициализация подключения к базе данных
        
        Args:
            db_config (dict, optional): Конфигурация подключения к базе данных
        """
        if db_config is None:
            # Настройки по умолчанию
            self.db_config = {
                'dbname': 'real_estate_database',
                'user': 'postgres',
                'password': 'ALENAalena0896',
                'host': '127.0.0.1',
                'port': '5432'
            }
        else:
            self.db_config = db_config
        
        self.conn = None
        self.cursor = None
    
    def connect(self):
        """Установка соединения с базой данных"""
        try:
            self.conn = psycopg2.connect(
                dbname=self.db_config['dbname'],
                user=self.db_config['user'],
                password=self.db_config['password'],
                host=self.db_config['host'],
                port=self.db_config['port']
            )
            self.cursor = self.conn.cursor()
            print("Успешное подключение к базе данных")
            return True
        except Exception as e:
            print(f"Ошибка подключения к базе данных: {e}")
            return False
    
    def disconnect(self):
        """Закрытие соединения с базой данных"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
        self.cursor = None
        self.conn = None
        print("Соединение с базой данных закрыто")
    
    def execute_query(self, query, params=None, fetchall=True, as_dict=False):
        """
        Выполнение SQL-запроса к базе данных
        
        Args:
            query (str): SQL-запрос
            params (tuple, optional): Параметры запроса
            fetchall (bool): Получить все результаты (True) или только один (False)
            as_dict (bool): Вернуть результаты в виде словаря (True) или кортежа (False)
        
        Returns:
            list/tuple: Результаты запроса
        """
        if not self.conn or self.conn.closed:
            if not self.connect():
                return None
        
        try:
            if as_dict:
                cursor = self.conn.cursor(cursor_factory=RealDictCursor)
            else:
                cursor = self.cursor
            
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            if query.strip().upper().startswith(('SELECT', 'WITH')):
                if fetchall:
                    result = cursor.fetchall()
                else:
                    result = cursor.fetchone()
                
                if as_dict:
                    cursor.close()
                
                return result
            else:
                self.conn.commit()
                return cursor.rowcount  # Количество затронутых строк
        
        except Exception as e:
            self.conn.rollback()
            print(f"Ошибка выполнения запроса: {e}")
            print(f"Запрос: {query}")
            if params:
                print(f"Параметры: {params}")
            return None
    
    def query_to_dataframe(self, query, params=None):
        """
        Выполнение запроса и получение результатов в виде DataFrame
        
        Args:
            query (str): SQL-запрос
            params (tuple, optional): Параметры запроса
        
        Returns:
            pandas.DataFrame: Результаты запроса в виде DataFrame
        """
        results = self.execute_query(query, params, fetchall=True, as_dict=True)
        
        if results:
            return pd.DataFrame(results)
        else:
            return pd.DataFrame()
    
    def get_database_info(self):
        """
        Получение общей информации о базе данных (количество записей в таблицах)
        
        Returns:
            dict: Информация о базе данных
        """
        info = {}
        
        # Таблицы для проверки
        tables = [
            'properties', 'districts', 'developers', 'financial_institutions', 
            'properties_districts', 'properties_developers', 'properties_institutions',
            'dates'
        ]
        
        for table in tables:
            query = f"SELECT COUNT(*) FROM {table}"
            count = self.execute_query(query, fetchall=False)
            
            if count:
                info[table] = count[0]
            else:
                info[table] = 0
        
        return info
    
    def get_property_by_id(self, property_id):
        """
        Получение информации о недвижимости по ID
        
        Args:
            property_id (int): ID объекта недвижимости
        
        Returns:
            dict: Информация об объекте недвижимости
        """
        query = """
        SELECT * FROM properties WHERE property_id = %s
        """
        
        result = self.execute_query(query, (property_id,), fetchall=False, as_dict=True)
        
        if not result:
            return None
        
        # Получаем районы
        districts_query = """
        SELECT d.district_name, d.district_type
        FROM properties_districts pd
        JOIN districts d ON pd.district_id = d.district_id
        WHERE pd.property_id = %s
        """
        
        districts = self.execute_query(districts_query, (property_id,), as_dict=True)
        result['districts'] = districts if districts else []
        
        # Получаем застройщиков
        developers_query = """
        SELECT d.developer_name, d.developer_type
        FROM properties_developers pd
        JOIN developers d ON pd.developer_id = d.developer_id
        WHERE pd.property_id = %s
        """
        
        developers = self.execute_query(developers_query, (property_id,), as_dict=True)
        result['developers'] = developers if developers else []
        
        # Получаем финансовые учреждения
        institutions_query = """
        SELECT fi.institution_name, fi.institution_type
        FROM properties_institutions pi
        JOIN financial_institutions fi ON pi.institution_id = fi.institution_id
        WHERE pi.property_id = %s
        """
        
        institutions = self.execute_query(institutions_query, (property_id,), as_dict=True)
        result['financial_institutions'] = institutions if institutions else []
        
        # Получаем даты
        dates_query = """
        SELECT date_type, date_value, day_of_week, month, year, quarter
        FROM dates
        WHERE property_id = %s
        """
        
        dates = self.execute_query(dates_query, (property_id,), as_dict=True)
        result['dates'] = dates if dates else []
        
        return result
    
    def search_properties(self, filters=None, limit=100, offset=0):
        """
        Поиск объектов недвижимости по фильтрам
        
        Args:
            filters (dict): Фильтры для поиска
            limit (int): Максимальное количество результатов
            offset (int): Смещение для пагинации
        
        Returns:
            pandas.DataFrame: Результаты поиска
        """
        query_parts = ["SELECT p.* FROM properties p"]
        where_clauses = []
        params = []
        
        if filters:
            # Обработка фильтров по основным свойствам
            if 'complex_name' in filters:
                where_clauses.append("p.complex_name ILIKE %s")
                params.append(f"%{filters['complex_name']}%")
            
            if 'region' in filters:
                where_clauses.append("p.region = %s")
                params.append(filters['region'])
            
            if 'min_floor' in filters:
                where_clauses.append("p.floor >= %s")
                params.append(filters['min_floor'])
            
            if 'max_floor' in filters:
                where_clauses.append("p.floor <= %s")
                params.append(filters['max_floor'])
            
            if 'property_type' in filters:
                where_clauses.append("p.property_type = %s")
                params.append(filters['property_type'])
            
            if 'room_type' in filters:
                where_clauses.append("p.room_type = %s")
                params.append(filters['room_type'])
            
            if 'finishing' in filters:
                where_clauses.append("p.finishing = %s")
                params.append(filters['finishing'])
            
            if 'min_price' in filters:
                where_clauses.append("p.price_per_sqm >= %s")
                params.append(filters['min_price'])
            
            if 'max_price' in filters:
                where_clauses.append("p.price_per_sqm <= %s")
                params.append(filters['max_price'])
            
            if 'mortgage' in filters:
                where_clauses.append("p.mortgage = %s")
                params.append(filters['mortgage'])
            
            # Фильтр по району
            if 'district' in filters:
                query_parts.append("JOIN properties_districts pd ON p.property_id = pd.property_id")
                query_parts.append("JOIN districts d ON pd.district_id = d.district_id")
                where_clauses.append("d.district_name = %s")
                params.append(filters['district'])
            
            # Фильтр по застройщику
            if 'developer' in filters:
                query_parts.append("JOIN properties_developers pd_dev ON p.property_id = pd_dev.property_id")
                query_parts.append("JOIN developers dev ON pd_dev.developer_id = dev.developer_id")
                where_clauses.append("dev.developer_name = %s")
                params.append(filters['developer'])
            
            # Фильтр по финансовому учреждению
            if 'institution' in filters:
                query_parts.append("JOIN properties_institutions pi ON p.property_id = pi.property_id")
                query_parts.append("JOIN financial_institutions fi ON pi.institution_id = fi.institution_id")
                where_clauses.append("fi.institution_name = %s")
                params.append(filters['institution'])
        
        # Добавляем WHERE условия, если они есть
        if where_clauses:
            query_parts.append("WHERE " + " AND ".join(where_clauses))
        
        # Добавляем LIMIT и OFFSET
        query_parts.append("ORDER BY p.property_id")
        query_parts.append(f"LIMIT {limit} OFFSET {offset}")
        
        # Собираем запрос
        query = " ".join(query_parts)
        
        # Выполняем запрос
        return self.query_to_dataframe(query, params)
    
    def get_districts(self, district_type=None):
        """
        Получение списка районов
        
        Args:
            district_type (str, optional): Тип района (Москва, СПб, МО, ЛО)
        
        Returns:
            pandas.DataFrame: Список районов
        """
        query = "SELECT * FROM districts"
        params = None
        
        if district_type:
            query += " WHERE district_type = %s"
            params = (district_type,)
        
        return self.query_to_dataframe(query, params)
    
    def get_developers(self, developer_type=None):
        """
        Получение списка застройщиков
        
        Args:
            developer_type (str, optional): Тип застройщика
        
        Returns:
            pandas.DataFrame: Список застройщиков
        """
        query = "SELECT * FROM developers"
        params = None
        
        if developer_type:
            query += " WHERE developer_type = %s"
            params = (developer_type,)
        
        return self.query_to_dataframe(query, params)
    
    def get_financial_institutions(self, institution_type=None):
        """
        Получение списка финансовых учреждений
        
        Args:
            institution_type (str, optional): Тип финансового учреждения
        
        Returns:
            pandas.DataFrame: Список финансовых учреждений
        """
        query = "SELECT * FROM financial_institutions"
        params = None
        
        if institution_type:
            query += " WHERE institution_type = %s"
            params = (institution_type,)
        
        return self.query_to_dataframe(query, params)
    
    def get_price_statistics(self, filters=None):
        """
        Получение статистики по ценам
        
        Args:
            filters (dict, optional): Фильтры для выборки
        
        Returns:
            dict: Статистика по ценам
        """
        query_parts = ["SELECT MIN(price_per_sqm) as min_price, MAX(price_per_sqm) as max_price, AVG(price_per_sqm) as avg_price, STDDEV(price_per_sqm) as std_price FROM properties p"]
        where_clauses = []
        params = []
        
        if filters:
            # Обработка фильтров (аналогично методу search_properties)
            if 'complex_name' in filters:
                where_clauses.append("p.complex_name ILIKE %s")
                params.append(f"%{filters['complex_name']}%")
            
            if 'region' in filters:
                where_clauses.append("p.region = %s")
                params.append(filters['region'])
            
            if 'property_type' in filters:
                where_clauses.append("p.property_type = %s")
                params.append(filters['property_type'])
            
            if 'room_type' in filters:
                where_clauses.append("p.room_type = %s")
                params.append(filters['room_type'])
            
            # Фильтр по району
            if 'district' in filters:
                query_parts.append("JOIN properties_districts pd ON p.property_id = pd.property_id")
                query_parts.append("JOIN districts d ON pd.district_id = d.district_id")
                where_clauses.append("d.district_name = %s")
                params.append(filters['district'])
            
            # Фильтр по застройщику
            if 'developer' in filters:
                query_parts.append("JOIN properties_developers pd_dev ON p.property_id = pd_dev.property_id")
                query_parts.append("JOIN developers dev ON pd_dev.developer_id = dev.developer_id")
                where_clauses.append("dev.developer_name = %s")
                params.append(filters['developer'])
        
        # Добавляем WHERE условия, если они есть
        if where_clauses:
            query_parts.append("WHERE " + " AND ".join(where_clauses))
        
        # Собираем запрос
        query = " ".join(query_parts)
        
        # Выполняем запрос
        result = self.execute_query(query, params, fetchall=False, as_dict=True)
        
        if result:
            # Преобразуем decimal в float для JSON-сериализации
            for key, value in result.items():
                if value is not None:
                    result[key] = float(value)
            return result
        else:
            return {
                'min_price': None,
                'max_price': None,
                'avg_price': None,
                'std_price': None
            }
    
    def get_price_distribution(self, filters=None, bins=10):
        """
        Получение распределения цен для визуализации
        
        Args:
            filters (dict, optional): Фильтры для выборки
            bins (int): Количество интервалов для гистограммы
        
        Returns:
            dict: Данные для построения гистограммы
        """
        # Получаем данные с применением фильтров
        query_parts = ["SELECT price_per_sqm FROM properties p"]
        where_clauses = []
        params = []
        
        if filters:
            # Обработка фильтров (аналогично методу search_properties)
            if 'complex_name' in filters:
                where_clauses.append("p.complex_name ILIKE %s")
                params.append(f"%{filters['complex_name']}%")
            
            if 'region' in filters:
                where_clauses.append("p.region = %s")
                params.append(filters['region'])
            
            if 'property_type' in filters:
                where_clauses.append("p.property_type = %s")
                params.append(filters['property_type'])
            
            if 'room_type' in filters:
                where_clauses.append("p.room_type = %s")
                params.append(filters['room_type'])
            
            # Фильтр по району
            if 'district' in filters:
                query_parts.append("JOIN properties_districts pd ON p.property_id = pd.property_id")
                query_parts.append("JOIN districts d ON pd.district_id = d.district_id")
                where_clauses.append("d.district_name = %s")
                params.append(filters['district'])
            
            # Фильтр по застройщику
            if 'developer' in filters:
                query_parts.append("JOIN properties_developers pd_dev ON p.property_id = pd_dev.property_id")
                query_parts.append("JOIN developers dev ON pd_dev.developer_id = dev.developer_id")
                where_clauses.append("dev.developer_name = %s")
                params.append(filters['developer'])
        
        # Добавляем WHERE условия, если они есть
        if where_clauses:
            query_parts.append("WHERE " + " AND ".join(where_clauses))
        
        # Собираем запрос
        query = " ".join(query_parts)
        
        # Выполняем запрос
        df = self.query_to_dataframe(query, params)
        
        if df.empty:
            return {
                'bins': [],
                'counts': [],
                'bin_edges': []
            }
        
        # Вычисляем гистограмму
        prices = df['price_per_sqm'].dropna().values
        hist, bin_edges = np.histogram(prices, bins=bins)
        
        # Преобразуем в формат для JSON
        return {
            'bins': [(bin_edges[i] + bin_edges[i+1]) / 2 for i in range(len(bin_edges)-1)],
            'counts': hist.tolist(),
            'bin_edges': bin_edges.tolist()
        }
    
    def get_price_by_district(self, district_type='Москва', limit=20):
        """
        Получение средних цен по районам
        
        Args:
            district_type (str): Тип района (Москва, СПб, МО, ЛО)
            limit (int): Максимальное количество районов
        
        Returns:
            pandas.DataFrame: Данные о ценах по районам
        """
        query = """
        SELECT d.district_name, 
               COUNT(p.property_id) as property_count,
               AVG(p.price_per_sqm) as avg_price,
               MIN(p.price_per_sqm) as min_price,
               MAX(p.price_per_sqm) as max_price
        FROM districts d
        JOIN properties_districts pd ON d.district_id = pd.district_id
        JOIN properties p ON pd.property_id = p.property_id
        WHERE d.district_type = %s AND p.price_per_sqm IS NOT NULL
        GROUP BY d.district_name
        HAVING COUNT(p.property_id) > 5
        ORDER BY avg_price DESC
        LIMIT %s
        """
        
        return self.query_to_dataframe(query, (district_type, limit))
    
    def get_price_by_room_type(self, filters=None):
        """
        Получение средних цен по типу комнатности
        
        Args:
            filters (dict, optional): Фильтры для выборки
        
        Returns:
            pandas.DataFrame: Данные о ценах по типу комнатности
        """
        query_parts = ["""
        SELECT p.room_type, 
               COUNT(p.property_id) as property_count,
               AVG(p.price_per_sqm) as avg_price,
               MIN(p.price_per_sqm) as min_price,
               MAX(p.price_per_sqm) as max_price
        FROM properties p
        """]
        
        where_clauses = ["p.room_type IS NOT NULL", "p.price_per_sqm IS NOT NULL"]
        params = []
        
        if filters:
            # Обработка фильтров
            if 'region' in filters:
                where_clauses.append("p.region = %s")
                params.append(filters['region'])
            
            if 'property_type' in filters:
                where_clauses.append("p.property_type = %s")
                params.append(filters['property_type'])
            
            # Фильтр по району
            if 'district' in filters:
                query_parts.append("JOIN properties_districts pd ON p.property_id = pd.property_id")
                query_parts.append("JOIN districts d ON pd.district_id = d.district_id")
                where_clauses.append("d.district_name = %s")
                params.append(filters['district'])
            
            # Фильтр по застройщику
            if 'developer' in filters:
                query_parts.append("JOIN properties_developers pd_dev ON p.property_id = pd_dev.property_id")
                query_parts.append("JOIN developers dev ON pd_dev.developer_id = dev.developer_id")
                where_clauses.append("dev.developer_name = %s")
                params.append(filters['developer'])
        
        # Добавляем WHERE условия
        query_parts.append("WHERE " + " AND ".join(where_clauses))
        query_parts.append("GROUP BY p.room_type")
        query_parts.append("ORDER BY CASE WHEN p.room_type = 'Студия' THEN 0 WHEN p.room_type = '1' THEN 1 WHEN p.room_type = '2' THEN 2 WHEN p.room_type = '3' THEN 3 WHEN p.room_type = '4+' THEN 4 ELSE 5 END")
        
        # Собираем запрос
        query = " ".join(query_parts)
        
        # Выполняем запрос
        return self.query_to_dataframe(query, params)
    
    def get_price_by_developer(self, filters=None, limit=20):
        """
        Получение средних цен по застройщикам
        
        Args:
            filters (dict, optional): Фильтры для выборки
            limit (int): Максимальное количество застройщиков
        
        Returns:
            pandas.DataFrame: Данные о ценах по застройщикам
        """
        query_parts = ["""
        SELECT dev.developer_name, 
               COUNT(p.property_id) as property_count,
               AVG(p.price_per_sqm) as avg_price,
               MIN(p.price_per_sqm) as min_price,
               MAX(p.price_per_sqm) as max_price
        FROM developers dev
        JOIN properties_developers pd_dev ON dev.developer_id = pd_dev.developer_id
        JOIN properties p ON pd_dev.property_id = p.property_id
        """]
        
        where_clauses = ["p.price_per_sqm IS NOT NULL"]
        params = []
        
        if filters:
            # Обработка фильтров
            if 'region' in filters:
                where_clauses.append("p.region = %s")
                params.append(filters['region'])
            
            if 'property_type' in filters:
                where_clauses.append("p.property_type = %s")
                params.append(filters['property_type'])
            
            if 'room_type' in filters:
                where_clauses.append("p.room_type = %s")
                params.append(filters['room_type'])
            
            # Фильтр по району
            if 'district' in filters:
                query_parts.append("JOIN properties_districts pd ON p.property_id = pd.property_id")
                query_parts.append("JOIN districts d ON pd.district_id = d.district_id")
                where_clauses.append("d.district_name = %s")
                params.append(filters['district'])
        
        # Добавляем WHERE условия
        query_parts.append("WHERE " + " AND ".join(where_clauses))
        query_parts.append("GROUP BY dev.developer_name")
        query_parts.append("HAVING COUNT(p.property_id) > 5")
        query_parts.append("ORDER BY avg_price DESC")
        query_parts.append(f"LIMIT {limit}")
        
        # Собираем запрос
        query = " ".join(query_parts)
        
        # Выполняем запрос
        return self.query_to_dataframe(query, params)
    
    def get_price_trend(self, date_type='registration', period='year', filters=None):
        """
        Получение тренда цен по времени
        
        Args:
            date_type (str): Тип даты (registration, encumbrance, ddu, sales_start, completion)
            period (str): Период агрегации (month, quarter, year)
            filters (dict, optional): Фильтры для выборки
        
        Returns:
            pandas.DataFrame: Данные о тренде цен
        """
        # Определяем группировку по периоду
        if period == 'month':
            date_group = "d.year, d.month"
            date_format = "to_char(d.date_value, 'YYYY-MM')"
        elif period == 'quarter':
            date_group = "d.year, d.quarter"
            date_format = "CASE WHEN d.quarter IS NOT NULL THEN CONCAT(d.year, '-Q', d.quarter) ELSE to_char(d.date_value, 'YYYY-\"Q\"Q') END"
        else:  # year
            date_group = "d.year"
            date_format = "CAST(d.year AS TEXT)"
        
        query_parts = [f"""
        SELECT {date_format} as period,
               {date_group} as period_parts,
               COUNT(p.property_id) as property_count,
               AVG(p.price_per_sqm) as avg_price,
               MIN(p.price_per_sqm) as min_price,
               MAX(p.price_per_sqm) as max_price
        FROM dates d
        JOIN properties p ON d.property_id = p.property_id
        """]
        
        where_clauses = [f"d.date_type = '{date_type}'", "p.price_per_sqm IS NOT NULL"]
        params = []
        
        if filters:
            # Обработка фильтров
            if 'region' in filters:
                where_clauses.append("p.region = %s")
                params.append(filters['region'])
            
            if 'property_type' in filters:
                where_clauses.append("p.property_type = %s")
                params.append(filters['property_type'])
            
            if 'room_type' in filters:
                where_clauses.append("p.room_type = %s")
                params.append(filters['room_type'])
            
            # Фильтр по району
            if 'district' in filters:
                query_parts.append("JOIN properties_districts pd ON p.property_id = pd.property_id")
                query_parts.append("JOIN districts d_dist ON pd.district_id = d_dist.district_id")
                where_clauses.append("d_dist.district_name = %s")
                params.append(filters['district'])
            
            # Фильтр по застройщику
            if 'developer' in filters:
                query_parts.append("JOIN properties_developers pd_dev ON p.property_id = pd_dev.property_id")
                query_parts.append("JOIN developers dev ON pd_dev.developer_id = dev.developer_id")
                where_clauses.append("dev.developer_name = %s")
                params.append(filters['developer'])
        
        # Добавляем WHERE условия
        query_parts.append("WHERE " + " AND ".join(where_clauses))
        query_parts.append(f"GROUP BY {date_group}, {date_format}")
        
        # Определяем сортировку в зависимости от периода
        if period == 'month':
            query_parts.append("ORDER BY d.year, d.month")
        elif period == 'quarter':
            query_parts.append("ORDER BY d.year, d.quarter")
        else:  # year
            query_parts.append("ORDER BY d.year")
        
        # Собираем запрос
        query = " ".join(query_parts)
        
        # Выполняем запрос
        return self.query_to_dataframe(query, params)
    
    def get_nearby_properties(self, latitude, longitude, radius_km=1, limit=50):
        """
        Получение объектов недвижимости рядом с указанными координатами
        
        Args:
            latitude (float): Широта
            longitude (float): Долгота
            radius_km (float): Радиус поиска в километрах
            limit (int): Максимальное количество результатов
        
        Returns:
            pandas.DataFrame: Объекты недвижимости в указанном радиусе
        """
        # Используем формулу гаверсинуса для расчета расстояния
        query = """
        SELECT 
            p.*,
            111.111 * 
            DEGREES(ACOS(LEAST(1.0, COS(RADIANS(%s)) * 
                   COS(RADIANS(latitude)) * 
                   COS(RADIANS(%s - longitude)) + 
                   SIN(RADIANS(%s)) * 
                   SIN(RADIANS(latitude))))) AS distance_km
        FROM properties p
        WHERE 
            latitude IS NOT NULL AND 
            longitude IS NOT NULL AND
            111.111 * 
            DEGREES(ACOS(LEAST(1.0, COS(RADIANS(%s)) * 
                   COS(RADIANS(latitude)) * 
                   COS(RADIANS(%s - longitude)) + 
                   SIN(RADIANS(%s)) * 
                   SIN(RADIANS(latitude))))) <= %s
        ORDER BY distance_km
        LIMIT %s
        """
        
        params = (latitude, longitude, latitude, latitude, longitude, latitude, radius_km, limit)
        
        return self.query_to_dataframe(query, params)
    
    def get_property_count_by_date(self, date_type='registration', period='month', filters=None):
        """
        Получение количества объектов недвижимости по датам
        
        Args:
            date_type (str): Тип даты (registration, encumbrance, ddu, sales_start, completion)
            period (str): Период агрегации (month, quarter, year)
            filters (dict, optional): Фильтры для выборки
        
        Returns:
            pandas.DataFrame: Данные о количестве объектов по датам
        """
        # Определяем группировку по периоду
        if period == 'month':
            date_group = "d.year, d.month"
            date_format = "to_char(d.date_value, 'YYYY-MM')"
        elif period == 'quarter':
            date_group = "d.year, d.quarter"
            date_format = "CASE WHEN d.quarter IS NOT NULL THEN CONCAT(d.year, '-Q', d.quarter) ELSE to_char(d.date_value, 'YYYY-\"Q\"Q') END"
        else:  # year
            date_group = "d.year"
            date_format = "CAST(d.year AS TEXT)"
        
        query_parts = [f"""
        SELECT {date_format} as period,
               {date_group} as period_parts,
               COUNT(p.property_id) as property_count
        FROM dates d
        JOIN properties p ON d.property_id = p.property_id
        """]
        
        where_clauses = [f"d.date_type = '{date_type}'"]
        params = []
        
        if filters:
            # Обработка фильтров
            if 'region' in filters:
                where_clauses.append("p.region = %s")
                params.append(filters['region'])
            
            if 'property_type' in filters:
                where_clauses.append("p.property_type = %s")
                params.append(filters['property_type'])
            
            if 'room_type' in filters:
                where_clauses.append("p.room_type = %s")
                params.append(filters['room_type'])
            
            # Фильтр по району
            if 'district' in filters:
                query_parts.append("JOIN properties_districts pd ON p.property_id = pd.property_id")
                query_parts.append("JOIN districts d_dist ON pd.district_id = d_dist.district_id")
                where_clauses.append("d_dist.district_name = %s")
                params.append(filters['district'])
            
            # Фильтр по застройщику
            if 'developer' in filters:
                query_parts.append("JOIN properties_developers pd_dev ON p.property_id = pd_dev.property_id")
                query_parts.append("JOIN developers dev ON pd_dev.developer_id = dev.developer_id")
                where_clauses.append("dev.developer_name = %s")
                params.append(filters['developer'])
        
        # Добавляем WHERE условия
        query_parts.append("WHERE " + " AND ".join(where_clauses))
        query_parts.append(f"GROUP BY {date_group}, {date_format}")
        
        # Определяем сортировку в зависимости от периода
        if period == 'month':
            query_parts.append("ORDER BY d.year, d.month")
        elif period == 'quarter':
            query_parts.append("ORDER BY d.year, d.quarter")
        else:  # year
            query_parts.append("ORDER BY d.year")
        
        # Собираем запрос
        query = " ".join(query_parts)
        
        # Выполняем запрос
        return self.query_to_dataframe(query, params)
    
    def get_similar_properties(self, property_id, limit=10):
        """
        Получение похожих объектов недвижимости
        
        Args:
            property_id (int): ID объекта недвижимости
            limit (int): Максимальное количество результатов
        
        Returns:
            pandas.DataFrame: Похожие объекты недвижимости
        """
        # Сначала получаем информацию об исходном объекте
        query = """
        SELECT * FROM properties WHERE property_id = %s
        """
        
        property_info = self.execute_query(query, (property_id,), fetchall=False, as_dict=True)
        
        if not property_info:
            return pd.DataFrame()
        
        # Теперь ищем похожие объекты
        query = """
        SELECT p.*,
               111.111 * 
               DEGREES(ACOS(LEAST(1.0, COS(RADIANS(%s)) * 
                      COS(RADIANS(latitude)) * 
                      COS(RADIANS(%s - longitude)) + 
                      SIN(RADIANS(%s)) * 
                      SIN(RADIANS(latitude))))) AS distance_km,
               ABS(price_per_sqm - %s) / %s AS price_diff_ratio
        FROM properties p
        WHERE p.property_id != %s
          AND p.room_type = %s
          AND p.price_per_sqm IS NOT NULL
          AND p.latitude IS NOT NULL
          AND p.longitude IS NOT NULL
        ORDER BY (
            -- Вес для расстояния (50%)
            0.5 * LEAST(1.0, 111.111 * 
                  DEGREES(ACOS(LEAST(1.0, COS(RADIANS(%s)) * 
                         COS(RADIANS(latitude)) * 
                         COS(RADIANS(%s - longitude)) + 
                         SIN(RADIANS(%s)) * 
                         SIN(RADIANS(latitude))))) / 5.0) +
            -- Вес для разницы в цене (50%)
            0.5 * LEAST(1.0, ABS(price_per_sqm - %s) / %s)
        ) 
        LIMIT %s
        """
        
        params = (
            property_info['latitude'], property_info['longitude'], property_info['latitude'],
            property_info['price_per_sqm'], property_info['price_per_sqm'],
            property_id, property_info['room_type'],
            property_info['latitude'], property_info['longitude'], property_info['latitude'],
            property_info['price_per_sqm'], property_info['price_per_sqm'],
            limit
        )
        
        return self.query_to_dataframe(query, params)

# Примеры использования
if __name__ == "__main__":
    # Создаем экземпляр класса
    db = RealEstateDB()
    
    # Подключаемся к базе данных
    if db.connect():
        # Получаем информацию о базе данных
        db_info = db.get_database_info()
        print("Информация о базе данных:")
        for table, count in db_info.items():
            print(f"  {table}: {count} записей")
        
        # Поиск объектов недвижимости
        filters = {
            'region': 'Москва',
            'room_type': '2',
            'min_price': 100000,
            'max_price': 300000
        }
        properties = db.search_properties(filters, limit=5)
        print(f"\nНайдено {len(properties)} объектов недвижимости:")
        print(properties[['complex_name', 'room_type', 'price_per_sqm']].head())
        
        # Статистика по ценам
        price_stats = db.get_price_statistics(filters)
        print("\nСтатистика по ценам:")
        for key, value in price_stats.items():
            print(f"  {key}: {value}")
        
        # Цены по районам
        district_prices = db.get_price_by_district(district_type='Москва', limit=5)
        print("\nЦены по районам Москвы:")
        print(district_prices[['district_name', 'property_count', 'avg_price']].head())
        
        # Цены по типу комнатности
        room_prices = db.get_price_by_room_type(filters={'region': 'Москва'})
        print("\nЦены по типу комнатности в Москве:")
        print(room_prices[['room_type', 'property_count', 'avg_price']])
        
        # Тренд цен
        price_trend = db.get_price_trend(date_type='registration', period='year', filters={'region': 'Москва'})
        print("\nТренд цен по годам в Москве:")
        print(price_trend[['period', 'property_count', 'avg_price']])
        
        # Закрываем соединение
        db.disconnect()
    else:
        print("Не удалось подключиться к базе данных.")
