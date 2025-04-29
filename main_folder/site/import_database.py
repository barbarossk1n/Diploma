import pandas as pd
import numpy as np
import psycopg2
from psycopg2.extras import execute_values
from sqlalchemy import create_engine
import re
from tqdm import tqdm
import os
import urllib.parse
from datetime import datetime
import gc  # Сборщик мусора
import time  # Для задержки при повторных попытках подключения
import sys  # Для вывода информации о прогрессе

# Установка кодировки вывода для Windows
if os.name == 'nt':
    os.system('chcp 65001')

def process_and_load_data(csv_path, db_config, chunk_size=10000):
    """
    Основная функция для обработки CSV-файла чанками и загрузки данных в PostgreSQL
    
    Параметры:
    csv_path (str): Путь к CSV-файлу
    db_config (dict): Конфигурация подключения к базе данных
    chunk_size (int): Размер чанка для чтения CSV
    """
    print(f"Загрузка данных из {csv_path}...")
    
    # Проверка существования файла
    if not os.path.exists(csv_path):
        print(f"Ошибка: Файл {csv_path} не найден")
        return
    
    # Определение кодировки файла
    encodings = ['utf-8', 'cp1251', 'latin1', 'iso-8859-1']
    file_encoding = None
    
    for encoding in encodings:
        try:
            with open(csv_path, 'r', encoding=encoding) as f:
                f.readline()  # Пробуем прочитать первую строку
            file_encoding = encoding
            print(f"Используемая кодировка файла: {file_encoding}")
            break
        except UnicodeDecodeError:
            continue
    
    if file_encoding is None:
        print("Ошибка: Не удалось определить кодировку файла")
        return
    
    # Подключение к PostgreSQL
    conn = None
    cursor = None
    engine = None
    
    try:
        # Простое подключение без использования строки DSN
        print("Подключение к PostgreSQL...")
        conn = psycopg2.connect(
            dbname=db_config['dbname'],
            user=db_config['user'],
            password=db_config['password'],
            host=db_config['host'],
            port=db_config['port']
        )
        
        # URL-кодирование пароля для SQLAlchemy
        encoded_password = urllib.parse.quote_plus(db_config['password'])
        
        # Создаем engine для SQLAlchemy с закодированным паролем
        engine_url = f"postgresql+psycopg2://{db_config['user']}:{encoded_password}@{db_config['host']}:{db_config['port']}/{db_config['dbname']}"
        engine = create_engine(engine_url)
        
        cursor = conn.cursor()
        
        # Тест подключения
        cursor.execute("SELECT version();")
        version = cursor.fetchone()[0]
        print(f"Подключение успешно! Версия PostgreSQL: {version}")
        
        # Создание таблиц в базе данных
        create_tables(cursor)
        conn.commit()
        
        # Сначала прочитаем только заголовки для определения категорий колонок
        print("Чтение заголовков файла...")
        headers = pd.read_csv(csv_path, nrows=0, encoding=file_encoding).columns.tolist()
        
        # Категоризация колонок
        categories = categorize_columns(headers)
        
        # Создание справочных таблиц на основе заголовков
        create_reference_tables_with_psycopg2(headers, categories, cursor, conn)
        
        # Получение ID из справочных таблиц для создания связей
        district_map = get_district_map(cursor)
        developer_map = get_developer_map(cursor)
        institution_map = get_institution_map(cursor)
        
        # Счетчик обработанных строк
        total_rows_processed = 0
        
        # Чтение и обработка CSV файла чанками
        print(f"Начинаем обработку CSV файла чанками размером {chunk_size}...")
        
        # Итератор для чтения чанками с указанием кодировки
        chunks = pd.read_csv(csv_path, chunksize=chunk_size, low_memory=False, encoding=file_encoding)
        
        for chunk_num, chunk in enumerate(chunks):
            print(f"Обработка чанка {chunk_num+1}, строки {total_rows_processed+1}-{total_rows_processed+len(chunk)}")
            
            # Обработка и загрузка основной таблицы properties
            properties_df = process_properties(chunk, categories)
            
            # Загрузка через psycopg2 вместо SQLAlchemy
            load_dataframe_to_postgres(properties_df, 'properties', cursor, conn)
            
            # Получение property_id для связей
            cursor.execute(f"SELECT property_id FROM properties ORDER BY property_id DESC LIMIT {len(chunk)}")
            property_ids = [row[0] for row in cursor.fetchall()]
            property_ids.reverse()  # Чтобы соответствовать порядку в чанке
            
            # Создание связей между объектами и справочными таблицами
            create_property_links(chunk, cursor, conn, categories, property_ids, 
                                district_map, developer_map, institution_map)
            
            # Обработка и загрузка дат
            dates_df = process_dates(chunk, property_ids)
            if not dates_df.empty:
                load_dataframe_to_postgres(dates_df, 'dates', cursor, conn)
            
            # Обновление счетчика
            total_rows_processed += len(chunk)
            
            # Принудительная очистка памяти
            del chunk
            gc.collect()
            
            # Вывод прогресса
            sys.stdout.write(f"\rОбработано строк: {total_rows_processed}")
            sys.stdout.flush()
        
        print(f"\nВсего обработано {total_rows_processed} строк")
        print("Данные успешно загружены в PostgreSQL")
        
    except Exception as e:
        print(f"Произошла ошибка: {e}")
        # Выводим полный стек ошибки для отладки
        import traceback
        traceback.print_exc()
        
    finally:
        # Закрытие соединений
        if cursor:
            cursor.close()
        if conn:
            conn.close()
        print("Соединения с базой данных закрыты")

def load_dataframe_to_postgres(df, table_name, cursor, conn):
    """
    Загрузка DataFrame в PostgreSQL с использованием psycopg2
    """
    if df.empty:
        print(f"Предупреждение: Пустой DataFrame для таблицы {table_name}")
        return
    
    try:
        # Метод 1: Построчная вставка с параметрами
        columns = df.columns.tolist()
        placeholders = ', '.join(['%s'] * len(columns))
        
        # Формируем SQL запрос
        query = f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({placeholders})"
        
        # Преобразуем DataFrame в список кортежей
        records = df.replace({np.nan: None}).to_records(index=False)
        data = [tuple(record) for record in records]
        
        # Вставляем данные пакетами
        batch_size = 1000
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            cursor.executemany(query, batch)
        
        conn.commit()
        print(f"Загружено {len(df)} строк в таблицу {table_name}")
        
    except Exception as e:
        conn.rollback()
        print(f"Ошибка при загрузке данных в таблицу {table_name}: {e}")
        
        # Если первый метод не сработал, пробуем второй метод: построчная вставка
        try:
            print("Пробуем построчную вставку...")
            
            # Вставляем по одной строке
            for _, row in df.iterrows():
                values = []
                for val in row:
                    if pd.isna(val):
                        values.append(None)
                    else:
                        values.append(val)
                
                # Создаем плейсхолдеры для параметров
                placeholders = ', '.join(['%s'] * len(values))
                
                # Формируем и выполняем запрос
                query = f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({placeholders})"
                cursor.execute(query, values)
            
            conn.commit()
            print(f"Загружено {len(df)} строк в таблицу {table_name} через построчную вставку")
            
        except Exception as row_error:
            conn.rollback()
            print(f"Ошибка при построчной вставке в таблицу {table_name}: {row_error}")
            raise

def create_reference_tables_with_psycopg2(headers, categories, cursor, conn):
    """
    Создание и заполнение справочных таблиц на основе заголовков с использованием psycopg2
    """
    print("Создание справочных таблиц...")
    
    # 1. Создание таблицы районов
    district_names = []
    district_types = []
    
    for col in categories['districts']:
        if col in headers:
            # Извлекаем название района из колонки
            district_name = col.replace('Район ', '')
            district_names.append(district_name)
            
            # Определяем тип района
            if '(СПб)' in district_name:
                district_type = 'СПб'
            elif '(ЛО)' in district_name:
                district_type = 'ЛО'
            elif any(term in district_name for term in ['п.', 'пос.', 'п.г.т.', 'дер.', 'с.', 'д.']):
                district_type = 'МО'
            else:
                district_type = 'Москва'
            
            district_types.append(district_type)
    
    if district_names:
        # Вставка районов пакетами
        batch_size = 100
        for i in range(0, len(district_names), batch_size):
            batch_names = district_names[i:i+batch_size]
            batch_types = district_types[i:i+batch_size]
            
            try:
                # Создаем список кортежей для массовой вставки
                values = [(name, type_) for name, type_ in zip(batch_names, batch_types)]
                
                # Используем execute_values для массовой вставки
                execute_values(
                    cursor,
                    "INSERT INTO districts (district_name, district_type) VALUES %s ON CONFLICT DO NOTHING",
                    values
                )
                conn.commit()
            except Exception as e:
                conn.rollback()
                print(f"Ошибка при вставке районов: {e}")
                
                # Пробуем вставлять по одному
                for name, type_ in zip(batch_names, batch_types):
                    try:
                        cursor.execute(
                            "INSERT INTO districts (district_name, district_type) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                            (name, type_)
                        )
                    except Exception as e2:
                        print(f"Ошибка при вставке района {name}: {e2}")
                conn.commit()
        
        print(f"Добавлено {len(district_names)} районов")
    
    # 2. Создание таблицы застройщиков
    developer_names = []
    developer_types = []
    
    for col in categories['developers']:
        if col in headers:
            developer_names.append(col)
            
            # Определяем тип застройщика (простая эвристика)
            if any(term in col for term in ['Group', 'Development', 'Девелопмент']):
                developer_type = 'Девелопер'
            elif any(term in col for term in ['СК', 'Строй', 'Строительн']):
                developer_type = 'Строительная компания'
            else:
                developer_type = 'Другое'
            
            developer_types.append(developer_type)
    
    if developer_names:
        # Вставка застройщиков пакетами
        batch_size = 100
        for i in range(0, len(developer_names), batch_size):
            batch_names = developer_names[i:i+batch_size]
            batch_types = developer_types[i:i+batch_size]
            
            try:
                # Создаем список кортежей для массовой вставки
                values = [(name, type_) for name, type_ in zip(batch_names, batch_types)]
                
                # Используем execute_values для массовой вставки
                execute_values(
                    cursor,
                    "INSERT INTO developers (developer_name, developer_type) VALUES %s ON CONFLICT DO NOTHING",
                    values
                )
                conn.commit()
            except Exception as e:
                conn.rollback()
                print(f"Ошибка при вставке застройщиков: {e}")
                
                # Пробуем вставлять по одному
                for name, type_ in zip(batch_names, batch_types):
                    try:
                        cursor.execute(
                            "INSERT INTO developers (developer_name, developer_type) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                            (name, type_)
                        )
                    except Exception as e2:
                        print(f"Ошибка при вставке застройщика {name}: {e2}")
                conn.commit()
        
        print(f"Добавлено {len(developer_names)} застройщиков")
    
    # 3. Создание таблицы финансовых учреждений
    institution_names = []
    institution_types = []
    
    for col in categories['financial_institutions']:
        if col in headers:
            institution_names.append(col)
            
            # Определяем тип финансового учреждения
            if any(term in col for term in ['Банк', 'БАНК']):
                institution_type = 'Банк'
            elif any(term in col for term in ['ПАО', 'АО']):
                institution_type = 'Акционерное общество'
            elif any(term in col for term in ['ООО']):
                institution_type = 'Общество с ограниченной ответственностью'
            else:
                institution_type = 'Другое'
            
            institution_types.append(institution_type)
    
    if institution_names:
        # Вставка финансовых учреждений пакетами
        batch_size = 100
        for i in range(0, len(institution_names), batch_size):
            batch_names = institution_names[i:i+batch_size]
            batch_types = institution_types[i:i+batch_size]
            
            try:
                # Создаем список кортежей для массовой вставки
                values = [(name, type_) for name, type_ in zip(batch_names, batch_types)]
                
                # Используем execute_values для массовой вставки
                execute_values(
                    cursor,
                    "INSERT INTO financial_institutions (institution_name, institution_type) VALUES %s ON CONFLICT DO NOTHING",
                    values
                )
                conn.commit()
            except Exception as e:
                conn.rollback()
                print(f"Ошибка при вставке финансовых учреждений: {e}")
                
                # Пробуем вставлять по одному
                for name, type_ in zip(batch_names, batch_types):
                    try:
                        cursor.execute(
                            "INSERT INTO financial_institutions (institution_name, institution_type) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                            (name, type_)
                        )
                    except Exception as e2:
                        print(f"Ошибка при вставке финансового учреждения {name}: {e2}")
                conn.commit()
        
        print(f"Добавлено {len(institution_names)} финансовых учреждений")
    
    print("Справочные таблицы созданы")

def create_tables(cursor):
    """Создание таблиц в базе данных"""
    
    # Таблица properties
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS properties (
        property_id SERIAL PRIMARY KEY,
        complex_name VARCHAR(255),
        region VARCHAR(100),
        floor INTEGER,
        property_type VARCHAR(100),
        encumbrance_duration INTEGER,
        encumbrance_type VARCHAR(100),
        assignment BOOLEAN,
        lots_bought INTEGER,
        legal_entity_buyer BOOLEAN,
        property_class VARCHAR(50),
        latitude DECIMAL(10, 8),
        longitude DECIMAL(11, 8),
        mortgage BOOLEAN,
        finishing VARCHAR(100),
        zone VARCHAR(100),
        completion_stage VARCHAR(100),
        frozen BOOLEAN,
        pd_issued BOOLEAN,
        room_type VARCHAR(50),
        studio BOOLEAN,
        price_per_sqm DECIMAL(12, 2)
    )
    """)
    
    # Таблица districts
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS districts (
        district_id SERIAL PRIMARY KEY,
        district_name VARCHAR(255) NOT NULL,
        district_type VARCHAR(50)
    )
    """)
    
    # Таблица properties_districts
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS properties_districts (
        property_id INTEGER REFERENCES properties(property_id),
        district_id INTEGER REFERENCES districts(district_id),
        PRIMARY KEY (property_id, district_id)
    )
    """)
    
    # Таблица developers
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS developers (
        developer_id SERIAL PRIMARY KEY,
        developer_name VARCHAR(255) NOT NULL,
        developer_type VARCHAR(100)
    )
    """)
    
    # Таблица properties_developers
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS properties_developers (
        property_id INTEGER REFERENCES properties(property_id),
        developer_id INTEGER REFERENCES developers(developer_id),
        PRIMARY KEY (property_id, developer_id)
    )
    """)
    
    # Таблица financial_institutions
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS financial_institutions (
        institution_id SERIAL PRIMARY KEY,
        institution_name VARCHAR(255) NOT NULL,
        institution_type VARCHAR(100)
    )
    """)
    
    # Таблица properties_institutions
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS properties_institutions (
        property_id INTEGER REFERENCES properties(property_id),
        institution_id INTEGER REFERENCES financial_institutions(institution_id),
        PRIMARY KEY (property_id, institution_id)
    )
    """)
    
    # Таблица dates
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS dates (
        date_id SERIAL PRIMARY KEY,
        property_id INTEGER REFERENCES properties(property_id),
        date_type VARCHAR(50),
        date_value DATE,
        day_of_week INTEGER,
        month INTEGER,
        year INTEGER,
        quarter INTEGER
    )
    """)

def categorize_columns(headers):
    """
    Категоризация колонок по типам на основе списка заголовков
    
    Возвращает словарь с категориями колонок
    """
    categories = {
        'properties': [
            'ЖК рус', 'Регион', 'Этаж', 'Тип помещения', 'Длительность обременения',
            'Тип обременения', 'Уступка', 'Купил лотов в ЖК', 'Покупатель ЮЛ',
            'Класс', 'lat', 'lng', 'Ипотека', 'Отделка', 'Зона', 'Стадия готовности в дату ДДУ',
            'Заморожен', 'Выпущена ПД', 'Тип Комнатности (обн.)', 'Студия', 'Цена кв.м'
        ],
        'date_columns': [
            'Дата регистрации день недели', 'Дата регистрации месяц', 'Дата регистрации год',
            'Дата обременения день недели', 'Дата обременения месяц', 'Дата обременения год',
            'Дата ДДУ день недели', 'Дата ДДУ месяц', 'Дата ДДУ год',
            'Год старта продаж К', 'Месяц старта продаж К',
            'Год срока сдачи', 'Квартал срока сдачи'
        ],
        'districts': [],
        'developers': [],
        'financial_institutions': []
    }
    
    # Определение районов (колонки, начинающиеся с "Район")
    for col in headers:
        if col.startswith('Район '):
            categories['districts'].append(col)
    
    # Определение застройщиков и финансовых учреждений
    remaining_cols = set(headers) - set(categories['properties']) - set(categories['date_columns']) - set(categories['districts'])
    
    # Простая эвристика: колонки с "Банк", "АО", "ПАО", "ООО" в названии считаем финансовыми учреждениями
    financial_pattern = re.compile(r'(Банк|БАНК|АО\s|\sАО|ПАО\s|\sПАО|ООО\s|\sООО)')
    
    for col in remaining_cols:
        if financial_pattern.search(col):
            categories['financial_institutions'].append(col)
        else:
            categories['developers'].append(col)
    
    print(f"Категоризировано колонок:")
    print(f"- Основные свойства: {len(categories['properties'])}")
    print(f"- Даты: {len(categories['date_columns'])}")
    print(f"- Районы: {len(categories['districts'])}")
    print(f"- Застройщики: {len(categories['developers'])}")
    print(f"- Финансовые учреждения: {len(categories['financial_institutions'])}")
    
    return categories

def get_district_map(cursor):
    """Получение словаря district_name -> district_id"""
    cursor.execute("SELECT district_id, district_name FROM districts")
    return {row[1]: row[0] for row in cursor.fetchall()}

def get_developer_map(cursor):
    """Получение словаря developer_name -> developer_id"""
    cursor.execute("SELECT developer_id, developer_name FROM developers")
    return {row[1]: row[0] for row in cursor.fetchall()}

def get_institution_map(cursor):
    """Получение словаря institution_name -> institution_id"""
    cursor.execute("SELECT institution_id, institution_name FROM financial_institutions")
    return {row[1]: row[0] for row in cursor.fetchall()}

def process_properties(df, categories):
    """
    Обработка и подготовка данных для таблицы properties
    """
    # Выбираем только нужные колонки, которые есть в данных
    properties_cols = [col for col in categories['properties'] if col in df.columns]
    properties_df = df[properties_cols].copy()
    
    # Переименовываем колонки для соответствия схеме базы данных
    column_mapping = {
        'ЖК рус': 'complex_name',
        'Регион': 'region',
        'Этаж': 'floor',
        'Тип помещения': 'property_type',
        'Длительность обременения': 'encumbrance_duration',
        'Тип обременения': 'encumbrance_type',
        'Уступка': 'assignment',
        'Купил лотов в ЖК': 'lots_bought',
        'Покупатель ЮЛ': 'legal_entity_buyer',
        'Класс': 'property_class',
        'lat': 'latitude',
        'lng': 'longitude',
        'Ипотека': 'mortgage',
        'Отделка': 'finishing',
        'Зона': 'zone',
        'Стадия готовности в дату ДДУ': 'completion_stage',
        'Заморожен': 'frozen',
        'Выпущена ПД': 'pd_issued',
        'Тип Комнатности (обн.)': 'room_type',
        'Студия': 'studio',
        'Цена кв.м': 'price_per_sqm'
    }
    
    # Переименовываем только те колонки, которые есть в данных
    rename_dict = {k: v for k, v in column_mapping.items() if k in properties_df.columns}
    properties_df = properties_df.rename(columns=rename_dict)
    
    # Преобразование типов данных
    for col in ['floor', 'encumbrance_duration', 'lots_bought']:
        if col in properties_df.columns:
            properties_df[col] = pd.to_numeric(properties_df[col], errors='coerce')
    
    for col in ['assignment', 'legal_entity_buyer', 'mortgage', 'frozen', 'pd_issued', 'studio']:
        if col in properties_df.columns:
            # Преобразование с учетом возможных разных типов данных
            properties_df[col] = properties_df[col].apply(lambda x: bool(x) if pd.notna(x) else None)
    
    if 'price_per_sqm' in properties_df.columns:
        properties_df['price_per_sqm'] = pd.to_numeric(properties_df['price_per_sqm'], errors='coerce')
    
    return properties_df

def create_property_links(df, cursor, conn, categories, property_ids, district_map, developer_map, institution_map):
    """
    Создание связей между объектами недвижимости и справочными таблицами
    """
    try:
        # Проверка длины property_ids
        if len(property_ids) < len(df):
            print(f"Предупреждение: количество property_ids ({len(property_ids)}) меньше количества строк в DataFrame ({len(df)})")
            # Используем только доступные property_ids
            df = df.iloc[:len(property_ids)]
        
        # Создаем связи для районов
        district_links = []
        for i, row in enumerate(df.itertuples()):
            property_id = property_ids[i]
            
            for col in categories['districts']:
                if col in df.columns:
                    district_name = col.replace('Район ', '')
                    if district_name in district_map:
                        # Если в данной строке значение для этого района = 1, создаем связь
                        try:
                            value = getattr(row, df.columns.get_loc(col) + 1)
                            if pd.notna(value) and value == 1:
                                district_links.append((property_id, district_map[district_name]))
                        except:
                            # Пропускаем ошибки в данных
                            continue
        
        # Создаем связи для застройщиков
        developer_links = []
        for i, row in enumerate(df.itertuples()):
            property_id = property_ids[i]
            
            for col in categories['developers']:
                if col in df.columns and col in developer_map:
                    # Если в данной строке значение для этого застройщика = 1, создаем связь
                    try:
                        value = getattr(row, df.columns.get_loc(col) + 1)
                        if pd.notna(value) and value == 1:
                            developer_links.append((property_id, developer_map[col]))
                    except:
                        continue
        
        # Создаем связи для финансовых учреждений
        institution_links = []
        for i, row in enumerate(df.itertuples()):
            property_id = property_ids[i]
            
            for col in categories['financial_institutions']:
                if col in df.columns and col in institution_map:
                    # Если в данной строке значение для этого учреждения = 1, создаем связь
                    try:
                        value = getattr(row, df.columns.get_loc(col) + 1)
                        if pd.notna(value) and value == 1:
                            institution_links.append((property_id, institution_map[col]))
                    except:
                        continue
        
        # Вставка связей пакетами с использованием execute_values
        
        # Вставка связей с районами
        if district_links:
            try:
                execute_values(
                    cursor,
                    "INSERT INTO properties_districts (property_id, district_id) VALUES %s ON CONFLICT DO NOTHING",
                    district_links,
                    page_size=1000
                )
                conn.commit()
                print(f"Добавлено {len(district_links)} связей с районами")
            except Exception as e:
                conn.rollback()
                print(f"Ошибка при вставке связей с районами: {e}")
                
                # Пробуем вставлять по одному
                for prop_id, dist_id in district_links:
                    try:
                        cursor.execute(
                            "INSERT INTO properties_districts (property_id, district_id) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                            (prop_id, dist_id)
                        )
                    except Exception as e2:
                        print(f"Ошибка при вставке связи района ({prop_id}, {dist_id}): {e2}")
                conn.commit()
        
        # Вставка связей с застройщиками
        if developer_links:
            try:
                execute_values(
                    cursor,
                    "INSERT INTO properties_developers (property_id, developer_id) VALUES %s ON CONFLICT DO NOTHING",
                    developer_links,
                    page_size=1000
                )
                conn.commit()
                print(f"Добавлено {len(developer_links)} связей с застройщиками")
            except Exception as e:
                conn.rollback()
                print(f"Ошибка при вставке связей с застройщиками: {e}")
                
                # Пробуем вставлять по одному
                for prop_id, dev_id in developer_links:
                    try:
                        cursor.execute(
                            "INSERT INTO properties_developers (property_id, developer_id) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                            (prop_id, dev_id)
                        )
                    except Exception as e2:
                        print(f"Ошибка при вставке связи застройщика ({prop_id}, {dev_id}): {e2}")
                conn.commit()
        
        # Вставка связей с финансовыми учреждениями
        if institution_links:
            try:
                execute_values(
                    cursor,
                    "INSERT INTO properties_institutions (property_id, institution_id) VALUES %s ON CONFLICT DO NOTHING",
                    institution_links,
                    page_size=1000
                )
                conn.commit()
                print(f"Добавлено {len(institution_links)} связей с финансовыми учреждениями")
            except Exception as e:
                conn.rollback()
                print(f"Ошибка при вставке связей с финансовыми учреждениями: {e}")
                
                # Пробуем вставлять по одному
                for prop_id, inst_id in institution_links:
                    try:
                        cursor.execute(
                            "INSERT INTO properties_institutions (property_id, institution_id) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                            (prop_id, inst_id)
                        )
                    except Exception as e2:
                        print(f"Ошибка при вставке связи финансового учреждения ({prop_id}, {inst_id}): {e2}")
                conn.commit()
    
    except Exception as e:
        print(f"Ошибка при создании связей: {e}")
        conn.rollback()

def process_dates(df, property_ids):
    """
    Обработка и подготовка данных для таблицы dates
    """
    # Проверка длины property_ids
    if len(property_ids) < len(df):
        print(f"Предупреждение: количество property_ids ({len(property_ids)}) меньше количества строк в DataFrame ({len(df)})")
        # Используем только доступные property_ids
        df = df.iloc[:len(property_ids)]
    
    date_records = []
    
    date_types = {
        'registration': ['Дата регистрации день недели', 'Дата регистрации месяц', 'Дата регистрации год'],
        'encumbrance': ['Дата обременения день недели', 'Дата обременения месяц', 'Дата обременения год'],
        'ddu': ['Дата ДДУ день недели', 'Дата ДДУ месяц', 'Дата ДДУ год'],
        'sales_start': ['Месяц старта продаж К', 'Год старта продаж К'],
        'completion': ['Квартал срока сдачи', 'Год срока сдачи']
    }
    
    for i, row in enumerate(df.itertuples()):
        property_id = property_ids[i]
        
        # Обработка дат регистрации, обременения и ДДУ
        for date_type, date_columns in date_types.items():
            if all(col in df.columns for col in date_columns):
                if date_type in ['registration', 'encumbrance', 'ddu']:
                    try:
                        day_of_week = getattr(row, df.columns.get_loc(date_columns[0]) + 1)
                        month = getattr(row, df.columns.get_loc(date_columns[1]) + 1)
                        year = getattr(row, df.columns.get_loc(date_columns[2]) + 1)
                        
                        # Создаем дату (для простоты берем первый день месяца)
                        if pd.notna(month) and pd.notna(year):
                            try:
                                date_value = datetime(int(year), int(month), 1).date()
                            except:
                                date_value = None
                        else:
                            date_value = None
                        
                        date_records.append({
                            'property_id': property_id,
                            'date_type': date_type,
                            'date_value': date_value,
                            'day_of_week': day_of_week if pd.notna(day_of_week) else None,
                            'month': month if pd.notna(month) else None,
                            'year': year if pd.notna(year) else None,
                            'quarter': None
                        })
                    except:
                        # Пропускаем ошибки в данных
                        continue
                
                elif date_type == 'sales_start':
                    try:
                        month = getattr(row, df.columns.get_loc(date_columns[0]) + 1) if date_columns[0] in df.columns else None
                        year = getattr(row, df.columns.get_loc(date_columns[1]) + 1) if date_columns[1] in df.columns else None
                        
                        if pd.notna(month) and pd.notna(year):
                            try:
                                date_value = datetime(int(year), int(month), 1).date()
                            except:
                                date_value = None
                        else:
                            date_value = None
                        
                        date_records.append({
                            'property_id': property_id,
                            'date_type': date_type,
                            'date_value': date_value,
                            'day_of_week': None,
                            'month': month if pd.notna(month) else None,
                            'year': year if pd.notna(year) else None,
                            'quarter': None
                        })
                    except:
                        continue
                
                elif date_type == 'completion':
                    try:
                        quarter = getattr(row, df.columns.get_loc(date_columns[0]) + 1) if date_columns[0] in df.columns else None
                        year = getattr(row, df.columns.get_loc(date_columns[1]) + 1) if date_columns[1] in df.columns else None
                        
                        if pd.notna(quarter) and pd.notna(year):
                            try:
                                month = ((int(quarter) - 1) * 3) + 1
                                date_value = datetime(int(year), month, 1).date()
                            except:
                                date_value = None
                        else:
                            date_value = None
                        
                        date_records.append({
                            'property_id': property_id,
                            'date_type': date_type,
                            'date_value': date_value,
                            'day_of_week': None,
                            'month': None,
                            'year': year if pd.notna(year) else None,
                            'quarter': quarter if pd.notna(quarter) else None
                        })
                    except:
                        continue
    
    if date_records:
        dates_df = pd.DataFrame(date_records)
        return dates_df
    else:
        return pd.DataFrame()

if __name__ == "__main__":
    # Конфигурация базы данных
    db_config = {
        'dbname': 'real_estate_database',
        'user': 'postgres',
        'password': 'ALENAalena0896',  # Используйте простой пароль без специальных символов
        'host': '127.0.0.1',     # Используем IP вместо 'localhost'
        'port': '5432'
    }
    
    # Путь к CSV-файлу
    csv_path = 'merge_TOTAL.csv'  # Укажите правильный путь к вашему файлу
    
    # Размер чанка - настройте в зависимости от доступной памяти
    chunk_size = 1000  # Уменьшен размер чанка для экономии памяти
    
    # Запуск процесса загрузки данных
    process_and_load_data(csv_path, db_config, chunk_size)
