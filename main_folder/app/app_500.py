import streamlit as st
import folium
from streamlit_folium import folium_static
from folium.plugins import Draw
import pandas as pd
import sqlite3
import os
import numpy as np
from sklearn.neighbors import BallTree, NearestNeighbors
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math
import matplotlib.pyplot as plt
from streamlit_folium import st_folium
import seaborn as sns
from datetime import datetime, date
import math
import time
import traceback
import joblib
from xgboost import XGBRegressor

# Настройка страницы и стилей
st.set_page_config(page_title="RestatEval", layout="wide")
sns.set_theme(style="whitegrid")
sns.set_context("talk")

# Настройка размера шрифта для всех графиков
plt.rcParams.update({
    'font.size': 6,
    'axes.titlesize': 8,
    'axes.labelsize': 6,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'legend.fontsize': 6
})

# Константы
DATA_FILE = "msk_test_updated.csv"
DB_PATH = "housing_database.db"
SPATIAL_INDEX_PATH = "spatial_index.pkl"
MODEL_PATH = "xgboost_model.pkl"
EARTH_RADIUS = 6371.0

def haversine_distance(lat1, lon1, lat2, lon2):
    try:
        lat1_rad = math.radians(float(lat1))
        lon1_rad = math.radians(float(lon1))
        lat2_rad = math.radians(float(lat2))
        lon2_rad = math.radians(float(lon2))
        
        dlon = lon2_rad - lon1_rad
        dlat = lat2_rad - lat1_rad
        
        a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        distance = EARTH_RADIUS * c
        
        return distance
    except Exception as e:
        st.warning(f"Ошибка при расчете расстояния: {e}")
        return 0.0

def create_haversine_function(conn):
    try:
        conn.create_function("haversine", 4, haversine_distance)
    except Exception as e:
        st.warning(f"Не удалось создать функцию haversine в SQLite: {e}")

def check_column_exists(cursor, table_name, column_name):
    try:
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = [info[1] for info in cursor.fetchall()]
        return column_name in columns
    except Exception as e:
        st.warning(f"Ошибка при проверке столбца {column_name}: {e}")
        return False
@st.cache_data
def create_housing_database(csv_path=DATA_FILE, db_path=DB_PATH):
    start_time = time.time()
    st.info("Загрузка и индексация данных. Это может занять некоторое время...")
    
    if not os.path.exists(csv_path):
        st.error(f"CSV файл не найден: {csv_path}")
        return None, None, None, None, None
    
    try:
        try:
            df = pd.read_csv(csv_path, usecols=['lat', 'lng', 'Класс', 'Цена со скидкой', 'Дата ДДУ год', 'Дата ДДУ месяц', 'ЖК', 'Этаж','Площадь', 'Год старта продаж К', 'Месяц старта продаж К', 'Отделка'], low_memory=False)
        except:
            df = pd.read_csv(csv_path, low_memory=False)
        
        required_columns = ['lat', 'lng']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            if 'lat' in missing_columns:
                lat_alternatives = [col for col in df.columns if 'lat' in col.lower() or 'широта' in col.lower()]
                if lat_alternatives:
                    df['lat'] = df[lat_alternatives[0]]
                else:
                    st.error("Не найдена колонка с широтой (lat)")
                    return None, None, None, None, None
            
            if 'lng' in missing_columns:
                lng_alternatives = [col for col in df.columns if 'lng' in col.lower() or 'lon' in col.lower() or 'долгота' in col.lower()]
                if lng_alternatives:
                    df['lng'] = df[lng_alternatives[0]]
                else:
                    st.error("Не найдена колонка с долготой (lng)")
                    return None, None, None, None, None
        
        price_cols = [col for col in df.columns if 'цена' in col.lower() or 'price' in col.lower()]
        for col in price_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
        df['lng'] = pd.to_numeric(df['lng'], errors='coerce')
        
        if 'Класс' not in df.columns:
            class_cols = [col for col in df.columns if 'класс' in col.lower()]
            if class_cols:
                df['Класс'] = df[class_cols[0]]
            else:
                df['Класс'] = 'Стандарт'
        
        if 'Дата ДДУ год' not in df.columns:
            year_cols = [col for col in df.columns if 'год' in col.lower()]
            if year_cols:
                df['Дата ДДУ год'] = pd.to_numeric(df[year_cols[0]], errors='coerce')
            else:
                df['Дата ДДУ год'] = 2024
        
        if 'Дата ДДУ месяц' not in df.columns:
            month_cols = [col for col in df.columns if 'месяц' in col.lower()]
            if month_cols:
                df['Дата ДДУ месяц'] = pd.to_numeric(df[month_cols[0]], errors='coerce')
            else:
                df['Дата ДДУ месяц'] = 4
        
        if 'Цена со скидкой' not in df.columns:
            price_cols = [col for col in df.columns if 'цена' in col.lower() or 'price' in col.lower()]
            if price_cols:
                df['Цена со скидкой'] = pd.to_numeric(df[price_cols[0]], errors='coerce')
            else:
                st.error("Не найдена колонка с ценой.")
                return None, None, None, None, None
                
        if 'ЖК' not in df.columns:
            complex_cols = [col for col in df.columns if 'жк' in col.lower() or 'комплекс' in col.lower()]
            if complex_cols:
                df['ЖК'] = df[complex_cols[0]]
            else:
                df['ЖК'] = 'Не указан'
        
        if 'Этаж' not in df.columns:
            floor_cols = [col for col in df.columns if 'этаж' in col.lower() or 'floor' in col.lower()]
            if floor_cols:
                df['Этаж'] = pd.to_numeric(df[floor_cols[0]], errors='coerce')
            else:
                df['Этаж'] = 5
        
        if 'Площадь' not in df.columns:
            area_cols = [col for col in df.columns if 'площадь' in col.lower() or 'area' in col.lower()]
            if area_cols:
                df['Площадь'] = pd.to_numeric(df[area_cols[0]], errors='coerce')
            else:
                df['Площадь'] = 60
        
        if 'Отделка' not in df.columns:
            finish_cols = [col for col in df.columns if 'отделк' in col.lower() or 'finish' in col.lower()]
            if finish_cols:
                df['Отделка'] = pd.to_numeric(df[finish_cols[0]], errors='coerce')
            else:
                df['Отделка'] = 0.0
        
        df['Отделка'] = pd.to_numeric(df['Отделка'], errors='coerce').fillna(0.0)
        if 'Год старта продаж К' not in df.columns:
            start_year_cols = [col for col in df.columns if 'год' in col.lower() and 'старт' in col.lower()]
            if start_year_cols:
                df['Год старта продаж К'] = pd.to_numeric(df[start_year_cols[0]], errors='coerce')
            else:
                df['Год старта продаж К'] = 2024
        
        if 'Месяц старта продаж К' not in df.columns:
            start_month_cols = [col for col in df.columns if 'месяц' in col.lower() and 'старт' in col.lower()]
            if start_month_cols:
                df['Месяц старта продаж К'] = pd.to_numeric(df[start_month_cols[0]], errors='coerce')
            else:
                df['Месяц старта продаж К'] = 1
        
        class_mapping = {
            1.0: "Эконом",
            2.0: "Комфорт",
            3.0: "Комфорт+",
            4.0: "Бизнес",
            5.0: "Элит"
        }
        df['Класс'] = df['Класс'].map(class_mapping).fillna(df['Класс'])
        
        if 'Тип Помещения' in df.columns:
            type_mapping = {
                0: "Апартаменты",
                1: "Квартира"
            }
            df['Тип Помещения'] = df['Тип Помещения'].map(type_mapping).fillna(df['Тип Помещения'])
        
        current_date = datetime.now()
        current_year = current_date.year
        current_month = current_date.month
        
        current_date_months = current_year * 12 + current_month
        
        df['start_date_months'] = df['Год старта продаж К'] * 12 + df['Месяц старта продаж К']
        
        df['months_on_sale'] = current_date_months - df['start_date_months']
        
        df_clean = df[df['months_on_sale'] <= 18].copy()
        
        df_clean = df_clean.dropna(subset=['lat', 'lng', 'Цена со скидкой']).copy()
        
        if len(df_clean) == 0:
            st.error("После фильтрации строк с пропущенными данными не осталось данных.")
            return None, None, None, None, None
        
        conn = sqlite3.connect(db_path)
        create_haversine_function(conn)
        df_clean.to_sql('properties', conn, if_exists='replace', index=False)
        
        cursor = conn.cursor()
        if check_column_exists(cursor, 'properties', 'lat'):
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_property_lat ON properties(lat)')
        if check_column_exists(cursor, 'properties', 'lng'):
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_property_lng ON properties(lng)')
        if check_column_exists(cursor, 'properties', 'Класс'):
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_property_class ON properties(Класс)')
        
        coords_rad = np.radians(df_clean[['lat', 'lng']].values)
        
        try:
            spatial_tree = BallTree(coords_rad, metric='haversine')
            
            xgboost_model = train_xgboost_model(df_clean)
            
            joblib.dump((spatial_tree, df_clean), SPATIAL_INDEX_PATH)
            joblib.dump(xgboost_model, MODEL_PATH)
            
            st.success("Пространственный индекс и модель XGBoost успешно созданы и сохранены.")
        except Exception as e:
            st.warning(f"Ошибка при создании моделей: {e}")
            spatial_tree = None
            xgboost_model = None
        
        conn.close()
        
        end_time = time.time()
        st.success(f"Данные успешно загружены и проиндексированы за {end_time - start_time:.2f} сек.")
        
        return df_clean.columns.tolist(), df_clean, spatial_tree, df_clean, xgboost_model
    
    except Exception as e:
        st.error(f"Ошибка при создании базы данных: {str(e)}")
        st.error(traceback.format_exc())
        return None, None, None, None, None

def train_xgboost_model(df):
    try:
        train_df = df.copy()
        
        features = ['lat', 'lng', 'Этаж', 'Площадь', 'Дата ДДУ год', 'Дата ДДУ месяц', 'months_on_sale', 'Отделка']
        categorical_features = ['Класс']
        
        available_features = [f for f in features if f in train_df.columns]
        available_cat_features = [f for f in categorical_features if f in train_df.columns]
        
        if not available_features:
            st.warning("Недостаточно числовых признаков для обучения модели")
            return None
        
        coords_rad = np.radians(train_df[['lat', 'lng']].values)
        spatial_tree = BallTree(coords_rad, metric='haversine')
        
        n_neighbors = 10
        
        train_df['neighbor_price_mean'] = np.nan
        train_df['neighbor_price_median'] = np.nan
        train_df['neighbor_price_std'] = np.nan
        train_df['neighbor_distance_mean'] = np.nan
        
        for idx, row in train_df.iterrows():
            if idx % 1000 == 0:
                st.info(f"Обрабатываем объект {idx} из {len(train_df)}")
            
            query_point_rad = np.radians([[row['lat'], row['lng']]])
            distances, indices = spatial_tree.query(query_point_rad, k=n_neighbors+1)
            
            neighbor_indices = indices[0][1:]
            neighbor_distances = distances[0][1:]
            
            if len(neighbor_indices) > 0:
                neighbors = train_df.iloc[neighbor_indices]
                
                train_df.at[idx, 'neighbor_price_mean'] = neighbors['Цена со скидкой'].mean()
                train_df.at[idx, 'neighbor_price_median'] = neighbors['Цена со скидкой'].median()
                train_df.at[idx, 'neighbor_price_std'] = neighbors['Цена со скидкой'].std()
                train_df.at[idx, 'neighbor_distance_mean'] = np.mean(neighbor_distances) * EARTH_RADIUS
        
        neighbor_features = ['neighbor_price_mean', 'neighbor_price_median', 
                            'neighbor_price_std', 'neighbor_distance_mean']
        
        all_features = available_features + neighbor_features
        
        X = train_df[all_features + available_cat_features].copy()
        y = train_df['Цена со скидкой']
        
        valid_indices = ~X.isna().any(axis=1) & ~y.isna()
        X = X[valid_indices]
        y = y[valid_indices]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        from sklearn.impute import SimpleImputer
        
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, all_features),
                ('cat', categorical_transformer, available_cat_features) if available_cat_features else ('cat', 'drop', [])
            ])
        
    
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            ))
        ])
        
        if len(X_train) < 10 or len(X_test) < 2:
            st.warning("Недостаточно данных для обучения модели")
            return None
        
        model.fit(X_train, y_train)
        
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        
        train_rmse = math.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = math.sqrt(mean_squared_error(y_test, y_test_pred))
        
        def calculate_mape(y_true, y_pred):
            y_true, y_pred = np.array(y_true), np.array(y_pred)
            mask = (y_true != 0)
            return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        
        train_mape = calculate_mape(y_train, y_train_pred)
        test_mape = calculate_mape(y_test, y_test_pred)
        
        model.metrics = {
            'train_score': train_score,
            'test_score': test_score,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mape': train_mape,
            'test_mape': test_mape,
            'feature_importance': None
        }
        
        # Получение важности признаков из XGBoost
        try:
            xgb_model = model.named_steps['regressor']
            feature_names = model.named_steps['preprocessor'].get_feature_names_out()
            importance = xgb_model.feature_importances_
            feature_importance = dict(zip(feature_names, importance))
            model.metrics['feature_importance'] = feature_importance
        except Exception as e:
            st.warning(f"Не удалось вычислить важность признаков: {str(e)}")
        
        return model
    except Exception as e:
        st.error(f"Ошибка при обучении модели XGBoost: {str(e)}")
        st.error(traceback.format_exc())
        return None
def evaluate_prediction_quality(predicted_price, actual_prices, neighbors):
    if len(actual_prices) == 0:
        return None
    
    from scipy import stats
    
    mean_price = np.mean(actual_prices)
    deviation_from_mean = abs(predicted_price - mean_price) / mean_price * 100
    
    median_price = np.median(actual_prices)
    deviation_from_median = abs(predicted_price - median_price) / median_price * 100
    
    percentile = stats.percentileofscore(actual_prices, predicted_price)
    
    closest_price_idx = np.argmin(np.abs(np.array(actual_prices) - predicted_price))
    closest_neighbor = neighbors.iloc[closest_price_idx]
    
    results = {
        'mean_price': mean_price,
        'median_price': median_price,
        'deviation_from_mean_percent': deviation_from_mean,
        'deviation_from_median_percent': deviation_from_median,
        'percentile': percentile,
        'closest_neighbor': closest_neighbor
    }
    
    return results

def find_nearest_neighbors_spatial(_lat, _lng, spatial_tree, spatial_df, n_neighbors=5, radius=10.0):
    try:
        query_point_rad = np.radians([[float(_lat), float(_lng)]])
        radius_rad = float(radius) / EARTH_RADIUS
        distances, indices = spatial_tree.query(query_point_rad, k=n_neighbors)
        
        if len(indices) == 0 or len(indices[0]) == 0:
            st.warning("Не найдено соседей. Пробуем увеличить радиус поиска.")
            distances, indices = spatial_tree.query(query_point_rad, k=min(100, len(spatial_df)))
        
        indices = indices[0]
        distances = distances[0]
        
        if len(indices) == 0:
            st.warning("Не найдено соседей даже при увеличенном радиусе поиска.")
            return pd.DataFrame()
        
        neighbors = spatial_df.iloc[indices].copy()
        neighbors['distance'] = distances * EARTH_RADIUS
        neighbors = neighbors.head(n_neighbors)
        
        neighbors['Номер квартиры'] = [f'Квартира {i+1}' for i in range(len(neighbors))]
        
        return neighbors
    
    except Exception as e:
        st.error(f"Ошибка при поиске ближайших соседей: {str(e)}")
        st.error(traceback.format_exc())
        return pd.DataFrame()

def find_similar_properties(predicted_price, _property_params, df, n_neighbors=5):
    try:
        df_search = df.copy()
        
        property_params_with_price = _property_params.copy()
        property_params_with_price['predicted_price'] = predicted_price
        
        features = ['lat', 'lng', 'Этаж', 'Площадь', 'Дата ДДУ год', 'Дата ДДУ месяц', 'months_on_sale', 'Отделка']
        cat_features = ['Класс']
        
        available_features = [f for f in features if f in df_search.columns and f in property_params_with_price]
        available_cat_features = [f for f in cat_features if f in df_search.columns and f in property_params_with_price]
        
        if not available_features:
            st.warning("Недостаточно общих признаков для поиска похожих объектов")
            return pd.DataFrame()
        X = df_search[available_features + available_cat_features].copy()
        
        query = pd.DataFrame([property_params_with_price], columns=property_params_with_price.keys())
        query_features = query[available_features + available_cat_features].copy()
        
        has_price_diff = False
        
        if 'Цена со скидкой' in df_search.columns:
            price_range = df_search['Цена со скидкой'].max() - df_search['Цена со скидкой'].min()
            if price_range > 0:
                X['price_diff'] = abs(df_search['Цена со скидкой'] - predicted_price) / price_range
                has_price_diff = True
                query_features['price_diff'] = 0
        
        from sklearn.impute import SimpleImputer
        
        numeric_imputer = SimpleImputer(strategy='median')
        categorical_imputer = SimpleImputer(strategy='most_frequent')
        
        numeric_features = [f for f in available_features if f != 'price_diff']
        if numeric_features:
            X_numeric = X[numeric_features]
            X_numeric_imputed = numeric_imputer.fit_transform(X_numeric)
            X_numeric_imputed_df = pd.DataFrame(X_numeric_imputed, columns=numeric_features, index=X.index)
            for col in numeric_features:
                X[col] = X_numeric_imputed_df[col]
            
            query_numeric = query_features[numeric_features]
            query_numeric_imputed = numeric_imputer.transform(query_numeric)
            query_numeric_imputed_df = pd.DataFrame(query_numeric_imputed, columns=numeric_features, index=query_features.index)
            for col in numeric_features:
                query_features[col] = query_numeric_imputed_df[col]
        
        if available_cat_features:
            X_cat = X[available_cat_features]
            X_cat_imputed = categorical_imputer.fit_transform(X_cat)
            X_cat_imputed_df = pd.DataFrame(X_cat_imputed, columns=available_cat_features, index=X.index)
            for col in available_cat_features:
                X[col] = X_cat_imputed_df[col]
            
            query_cat = query_features[available_cat_features]
            query_cat_imputed = categorical_imputer.transform(query_cat)
            query_cat_imputed_df = pd.DataFrame(query_cat_imputed, columns=available_cat_features, index=query_features.index)
            for col in available_cat_features:
                query_features[col] = query_cat_imputed_df[col]
        
        numeric_scaler = StandardScaler()
        if numeric_features:
            X[numeric_features] = numeric_scaler.fit_transform(X[numeric_features])
            query_features[numeric_features] = numeric_scaler.transform(query_features[numeric_features])
        
        if available_cat_features:
            X_cat_encoded = pd.get_dummies(X[available_cat_features], drop_first=False)
            query_cat_encoded = pd.get_dummies(query_features[available_cat_features], drop_first=False)
            
            missing_cols = set(X_cat_encoded.columns) - set(query_cat_encoded.columns)
            for col in missing_cols:
                query_cat_encoded[col] = 0
            query_cat_encoded = query_cat_encoded[X_cat_encoded.columns]
            
            X = X.drop(columns=available_cat_features)
            X = pd.concat([X, X_cat_encoded], axis=1)
            
            query_features = query_features.drop(columns=available_cat_features)
            query_features = pd.concat([query_features, query_cat_encoded], axis=1)
        
        if has_price_diff and 'price_diff' in X.columns:
            X['price_diff'] = X['price_diff'] * 2
        
        X = X.fillna(0)
        query_features = query_features.fillna(0)
        
        knn = NearestNeighbors(n_neighbors=min(n_neighbors, len(X)), metric='euclidean')
        knn.fit(X)
        
        distances, indices = knn.kneighbors(query_features)
        
        neighbors = df_search.iloc[indices[0]].copy()
        neighbors['distance'] = distances[0]
        
        neighbors['Номер квартиры'] = [f'Квартира {i+1}' for i in range(len(neighbors))]
        
        for i, row in neighbors.iterrows():
            neighbors.at[i, 'geo_distance'] = haversine_distance(
                _property_params['lat'], _property_params['lng'],
                row['lat'], row['lng']
            )
        
        return neighbors
    
    except Exception as e:
        st.error(f"Ошибка при поиске похожих объектов: {str(e)}")
        st.error(traceback.format_exc())
        return pd.DataFrame()

def predict_price_with_model(_property_params, model, df, neighbors=None):
    try:
        if model is None:
            st.warning("Модель XGBoost не загружена")
            return None
        
        query = pd.DataFrame([_property_params])
        
        num_features = model.named_steps['preprocessor'].transformers_[0][2]
        
        if neighbors is not None and len(neighbors) > 0:
            price_column = 'Цена со скидкой'
            if price_column in neighbors.columns:
                query['neighbor_price_mean'] = neighbors[price_column].mean()
                query['neighbor_price_median'] = neighbors[price_column].median()
                query['neighbor_price_std'] = neighbors[price_column].std()
                
                if 'distance' in neighbors.columns:
                    query['neighbor_distance_mean'] = neighbors['distance'].mean()
        
        missing_features = [f for f in num_features if f not in query.columns]
        if missing_features:
            for feature in missing_features:
                query[feature] = np.nan
        
        try:
            predicted_price = model.predict(query)[0]
            return max(0, predicted_price)
        except Exception as inner_e:
            st.warning(f"Ошибка при предсказании с помощью XGBoost: {inner_e}")
            return None
    
    except Exception as e:
        st.error(f"Ошибка при предсказании цены XGBoost: {str(e)}")
        st.error(traceback.format_exc())
        return None

@st.cache_data
def plot_neighbors_price_trends(prediction_info):
    prediction_info_str = str(prediction_info)
    
    if not prediction_info or 'neighbors' not in prediction_info:
        return None
    
    neighbors = prediction_info['neighbors']
    price_column = prediction_info.get('price_column', 'Цена со скидкой')
    predicted_price = prediction_info['predicted_price']
    
    df_viz = pd.DataFrame(neighbors)
    
    if price_column not in df_viz.columns:
        return None
    
    df_viz[price_column] = pd.to_numeric(df_viz[price_column], errors='coerce')
    df_viz = df_viz.dropna(subset=[price_column])
    
    if len(df_viz) < 2:
        return None
    
    if 'distance' not in df_viz.columns and 'distances' in prediction_info:
        distances = prediction_info['distances']
        if len(distances) != len(df_viz):
            distances = distances[:len(df_viz)]
        df_viz['distance'] = distances
    
    distance_col = 'geo_distance' if 'geo_distance' in df_viz.columns else 'distance'
    
    if distance_col in df_viz.columns:
        df_viz = df_viz.sort_values(distance_col)
    
    plt.figure(figsize=(6, 4))
    
    sns.scatterplot(
        data=df_viz, 
        x=distance_col, 
        y=price_column, 
        hue='Класс' if 'Класс' in df_viz.columns else None,
        s=40, 
        alpha=0.7,
        palette='viridis'
    )
    
    if len(df_viz) > 1:
        sns.regplot(
            data=df_viz, 
            x=distance_col, 
            y=price_column, 
            scatter=False, 
            line_kws={"color": "red", "linestyle": "--", "linewidth": 1.5}
        )
    
    plt.axhline(
        y=predicted_price, 
        color='#FF5733', 
        linestyle='-', 
        linewidth=1.5,
        label=f'Предсказанная цена: {predicted_price:.2f}'
    )
    
    for i, row in df_viz.iterrows():
        plt.annotate(
            row.get('Номер квартиры', f'Квартира {i+1}'),
            (row[distance_col], row[price_column]),
            xytext=(4, 4),
            textcoords='offset points',
            fontsize=5,  
            alpha=0.8
        )
    
    distance_label = "Расстояние от выбранной точки (км)" if distance_col == 'geo_distance' or distance_col == 'distance' else "Различие между объектами"
    plt.title('Тенденция цены по похожим объектам', fontsize=8)
    plt.xlabel(distance_label, fontsize=6)
    plt.ylabel('Цена', fontsize=6)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=6, loc='best')
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    
    sns.despine()
    plt.tight_layout()
    
    return plt.gcf()

def plot_price_per_sqm(prediction_info):
    if not prediction_info or 'neighbors' not in prediction_info:
        return None
    
    neighbors = prediction_info['neighbors']
    df_viz = pd.DataFrame(neighbors)
    
    price_column = prediction_info.get('price_column', 'Цена со скидкой')
    if price_column not in df_viz.columns or 'Площадь' not in df_viz.columns:
        return None
    
    df_viz[price_column] = pd.to_numeric(df_viz[price_column], errors='coerce')
    df_viz['Площадь'] = pd.to_numeric(df_viz['Площадь'], errors='coerce')
    df_viz = df_viz.dropna(subset=[price_column, 'Площадь'])
    
    if len(df_viz) < 2:
        return None
    
    df_viz['Цена за м²'] = df_viz[price_column] / df_viz['Площадь']
    
    plt.figure(figsize=(6, 4))
    
    if 'Класс' in df_viz.columns:
        sns.scatterplot(
            data=df_viz,
            x='Площадь',
            y='Цена за м²',
            hue='Класс',
            size='distance' if 'distance' in df_viz.columns else None,
            sizes=(20, 80),  
            alpha=0.7,
            palette='viridis'
        )
    else:
        sns.scatterplot(
            data=df_viz,
            x='Площадь',
            y='Цена за м²',
            size='distance' if 'distance' in df_viz.columns else None,
            sizes=(20, 80),  
            alpha=0.7,
            color='#1f77b4'
        )
    
    sns.regplot(
        data=df_viz,
        x='Площадь',
        y='Цена за м²',
        scatter=False,
        line_kws={"color": "red", "linestyle": "--", "linewidth": 1.5}
    )
    
    for i, row in df_viz.iterrows():
        plt.annotate(
            row.get('Номер квартиры', f'Квартира {i+1}'),
            (row['Площадь'], row['Цена за м²']),
            xytext=(4, 4),
            textcoords='offset points',
            fontsize=5, 
            alpha=0.8
        )
    
    plt.title('Зависимость цены за квадратный метр от площади', fontsize=8)
    plt.xlabel('Площадь (м²)', fontsize=6)
    plt.ylabel('Цена за м²', fontsize=6)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    sns.despine()
    
    if plt.gca().get_legend() is not None:
        plt.legend(fontsize=6)
    
    return plt.gcf()

def plot_price_by_floor(prediction_info):
    if not prediction_info or 'neighbors' not in prediction_info:
        return None
    
    neighbors = prediction_info['neighbors']
    df_viz = pd.DataFrame(neighbors)
    
    price_column = prediction_info.get('price_column', 'Цена со скидкой')
    if price_column not in df_viz.columns or 'Этаж' not in df_viz.columns:
        return None
    
    df_viz[price_column] = pd.to_numeric(df_viz[price_column], errors='coerce')
    df_viz['Этаж'] = pd.to_numeric(df_viz['Этаж'], errors='coerce')
    df_viz = df_viz.dropna(subset=[price_column, 'Этаж'])
    
    if len(df_viz) < 2:
        return None
    
    plt.figure(figsize=(6, 4))
    
    sns.scatterplot(
        data=df_viz,
        x='Этаж',
        y=price_column,
        hue='Класс' if 'Класс' in df_viz.columns else None,
        size='Площадь' if 'Площадь' in df_viz.columns else None,
        sizes=(20, 80),  
        alpha=0.7,
        palette='viridis'
    )
    
    sns.regplot(
        data=df_viz,
        x='Этаж',
        y=price_column,
        scatter=False,
        line_kws={"color": "red", "linestyle": "--", "linewidth": 1.5}
    )
    
    for i, row in df_viz.iterrows():
        plt.annotate(
            row.get('Номер квартиры', f'Квартира {i+1}'),
            (row['Этаж'], row[price_column]),
            xytext=(4, 4),
            textcoords='offset points',
            fontsize=5,  
            alpha=0.8
        )
    
    plt.title('Зависимость цены от этажа', fontsize=8)
    plt.xlabel('Этаж', fontsize=6)
    plt.ylabel('Цена', fontsize=6)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    sns.despine()
    
    if plt.gca().get_legend() is not None:
        plt.legend(fontsize=6)
    
    return plt.gcf()

@st.cache_data
def create_interactive_map(df):
    try:
        center_lat = df['lat'].mean() if 'lat' in df.columns else 55.7558
        center_lng = df['lng'].mean() if 'lng' in df.columns else 37.6176
        
        m = folium.Map(
            location=[55.7558, 37.6176], 
            zoom_start=11,
            control_scale=True,
            attributionControl=False
        )
        
        draw = Draw(
            draw_options={
                'polyline': False,
                'rectangle': False,
                'polygon': False,
                'circle': False,
                'marker': True,
                'circlemarker': False,
            },
            edit_options={'edit': False}
        )
        draw.add_to(m)
        
        custom_attr_html = """
        <div id="custom-attribution" style="
            position: absolute;
            bottom: 0;
            right: 0;
            background-color: white;
            padding: 4px 12px;
            border: 1px solid #4a4a4a;
            border-radius: 3px;
            font-family: Arial, sans-serif;
            font-weight: bold;
            font-size: 14px;
            margin: 0 10px 10px 0;
            z-index: 1000;
            min-width: 300px;
            text-align: center;
        ">RestatEval</div>
        """
        
        m.get_root().html.add_child(folium.Element(custom_attr_html))
        
        if 'lat' not in df.columns or 'lng' not in df.columns:
            return m
        
        price_column = 'Цена со скидкой'
        if price_column not in df.columns:
            for col in df.columns:
                if 'price' in col.lower() or 'цена' in col.lower():
                    price_column = col
                    break
        
        df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
        df['lng'] = pd.to_numeric(df['lng'], errors='coerce')
        df_map = df.dropna(subset=['lat', 'lng'])
        
        sample_size = min(50, len(df_map))
        sample_df = df_map.sample(sample_size) if len(df_map) > sample_size else df_map
        
        for idx, row in sample_df.iterrows():
            if pd.notna(row['lat']) and pd.notna(row['lng']):
                class_val = row.get('Класс', 'Н/Д') if 'Класс' in df.columns else 'Н/Д'
                year_val = row.get('Дата ДДУ год', 'Н/Д') if 'Дата ДДУ год' in df.columns else 'Н/Д'
                month_val = row.get('Дата ДДУ месяц', 'Н/Д') if 'Дата ДДУ месяц' in df.columns else 'Н/Д'
                
                if price_column in row and pd.notna(row[price_column]) and isinstance(row[price_column], (int, float)):
                    price_display = f"{row[price_column]:.2f}"
                else:
                    price_display = "Н/Д"
                
                popup_text = f"""
                    <b>Объект #{idx}</b><br>
                    Класс: {class_val}<br>
                    Цена: {price_display}<br>
                    Год: {year_val}<br>
                    Месяц: {month_val}
                """
                folium.Marker(
                    location=[row['lat'], row['lng']],
                    popup=folium.Popup(popup_text, max_width=300),
                    icon=folium.Icon(color='blue', icon='home')
                ).add_to(m)
        
        return m
    except Exception as e:
        st.error(f"Ошибка при создании карты: {str(e)}")
        return folium.Map(location=[55.7558, 37.6176], zoom_start=11)

@st.cache_resource
def load_spatial_index_and_model():
    try:
        spatial_tree = None
        spatial_df = None
        xgboost_model = None
        
        if os.path.exists(SPATIAL_INDEX_PATH):
            try:
                spatial_tree, spatial_df = joblib.load(SPATIAL_INDEX_PATH)
                st.success("Пространственный индекс успешно загружен из файла.")
            except Exception as e:
                st.warning(f"Не удалось загрузить существующий пространственный индекс: {e}")
                try:
                    os.remove(SPATIAL_INDEX_PATH)
                    st.info("Поврежденный файл пространственного индекса удален. Будет создан новый.")
                except:
                    pass
        
        if os.path.exists(MODEL_PATH):
            try:
                xgboost_model = joblib.load(MODEL_PATH)
                st.success("Модель XGBoost успешно загружена из файла.")
            except Exception as e:
                st.warning(f"Не удалось загрузить существующую модель XGBoost: {e}")
                try:
                    os.remove(MODEL_PATH)
                    st.info("Поврежденный файл модели удален. Будет создана новая модель.")
                except:
                    pass
        
        return spatial_tree, spatial_df, xgboost_model
    except Exception as e:
        st.error(f"Ошибка при загрузке моделей: {str(e)}")
        return None, None, None

@st.cache_data
def load_dataset(file_path):
    try:
        try:
            df = pd.read_csv(file_path, usecols=['lat', 'lng', 'Класс', 'Цена со скидкой', 'Дата ДДУ год', 'Дата ДДУ месяц', 'ЖК', 'Этаж', 'Площадь', 'Год старта продаж К', 'Месяц старта продаж К', 'Отделка'], low_memory=False)
        except:
            df = pd.read_csv(file_path, low_memory=False)
            
        class_mapping = {
            1.0: "Эконом",
            2.0: "Комфорт",
            3.0: "Комфорт+",
            4.0: "Бизнес",
            5.0: "Элит"
        }
        df['Класс'] = df['Класс'].map(class_mapping).fillna(df['Класс'])
        
        if 'Тип Помещения' in df.columns:
            type_mapping = {
                0: "Апартаменты",
                1: "Квартира"
            }
            df['Тип Помещения'] = df['Тип Помещения'].map(type_mapping).fillna(df['Тип Помещения'])
        
        if 'Отделка' in df.columns:
            df['Отделка'] = pd.to_numeric(df['Отделка'], errors='coerce').fillna(0.0)
        
        current_date = datetime.now()
        current_year = current_date.year
        current_month = current_date.month
        
        current_date_months = current_year * 12 + current_month
        
        df['start_date_months'] = df['Год старта продаж К'] * 12 + df['Месяц старта продаж К']
        
        df['months_on_sale'] = current_date_months - df['start_date_months']
        
        df = df[df['months_on_sale'] <= 18].copy()
            
        return df
    except Exception as e:
        st.error(f"Ошибка при загрузке данных: {str(e)}")
        return None

def set_custom_css():
    st.markdown("""
    <style>
    body {
        font-size: 18px !important;
    }
    
    label, .stSelectbox label, .stSlider label, .stNumberInput label {
        font-size: 18px !important;
    }
    
    input, select, .stSelectbox > div, .stNumberInput > div {
        font-size: 18px !important;
    }
    
    button, .stButton > button {
        font-size: 18px !important;
    }
    
    .stTabs [role="tab"] {
        font-size: 26px !important;
        font-weight: bold !important;
    }
    
    h1 {
        font-size: 30px !important;
    }
    
    h2 {
        font-size: 24px !important;
    }
    
    h3 {
        font-size: 18px !important;
    }
    
    p, div, span {
        font-size: 18px !important;
    }
    
    .stAlert > div {
        font-size: 18px !important;
    }
    
    .dataframe, .dataframe th, .dataframe td {
        font-size: 20px !important;
    }
    
    .streamlit-container, .streamlit-container div, .streamlit-container label, 
    .streamlit-container input, .streamlit-container button, .streamlit-container select, 
    .streamlit-container textarea {
        font-size: 18px !important;
    }
    
    * {
        font-size: 18px !important;
        font-family: 'Arial', sans-serif !important;
    }
    
    .stTabs [role="tab"] {
        font-size: 18px !important;
    }
    
    h1, h1 * {
        font-size: 30px !important;
    }
    
    h2, h2 * {
        font-size: 22px !important;
    }
    
    .stButton > button {
        font-size: 18px !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
def predict_price_and_find_neighbors(_property_params, df, spatial_tree, spatial_df, xgboost_model, n_neighbors=5):
    """Предсказывает цену, сначала находя ближайших соседей, а затем используя их для модели"""
    try:
        if df is None or len(df) == 0:
            st.error("Датафрейм не содержит данных для анализа")
            return 0.0, {}

        # Шаг 1: Находим ближайших соседей по пространственному индексу
        start_time = time.time()
        neighbors = find_nearest_neighbors_spatial(
            _property_params['lat'], 
            _property_params['lng'], 
            spatial_tree, 
            spatial_df, 
            n_neighbors=max(n_neighbors, 10),
            radius=10.0
        )
        
        if len(neighbors) == 0:
            st.warning("Не найдено соседей. Пробуем увеличить радиус поиска.")
            neighbors = find_nearest_neighbors_spatial(
                _property_params['lat'], 
                _property_params['lng'], 
                spatial_tree, 
                spatial_df, 
                n_neighbors=max(n_neighbors, 10),
                radius=20.0
            )
        
        if len(neighbors) < 3:
            st.error("Недостаточно соседей для надежного предсказания.")
            return 0.0, {}
        
        # Шаг 2: Создаем вектор признаков на основе найденных соседей
        price_column = 'Цена со скидкой'
        
        # Базовые статистики по ценам соседей
        neighbor_prices = neighbors[price_column].values
        mean_price = np.mean(neighbor_prices)
        median_price = np.median(neighbor_prices)
        min_price = np.min(neighbor_prices)
        max_price = np.max(neighbor_prices)
        price_range = max_price - min_price
        price_std = np.std(neighbor_prices)
        
        # Статистики по другим параметрам соседей
        neighbor_features = {}
        numerical_cols = ['Этаж', 'Площадь', 'Дата ДДУ год', 'Дата ДДУ месяц', 'distance', 'months_on_sale', 'Отделка']
        for col in numerical_cols:
            if col in neighbors.columns:
                neighbor_features[f'{col}_mean'] = np.mean(neighbors[col])
                neighbor_features[f'{col}_median'] = np.median(neighbors[col])
                neighbor_features[f'{col}_std'] = np.std(neighbors[col])
        
        # Категориальные признаки - берем моду
        categorical_cols = ['Класс', 'ЖК']
        for col in categorical_cols:
            if col in neighbors.columns:
                most_common = neighbors[col].mode()[0] if not neighbors[col].empty else "Не определено"
                neighbor_features[f'{col}_mode'] = most_common
                
                if col in _property_params:
                    same_value_ratio = sum(neighbors[col] == _property_params[col]) / len(neighbors)
                    neighbor_features[f'{col}_same_ratio'] = same_value_ratio
        
        # Шаг 3: Объединяем признаки объекта с признаками соседей
        combined_features = _property_params.copy()
        combined_features.update(neighbor_features)
        
        # Добавляем статистики по ценам
        combined_features['neighbor_price_mean'] = mean_price
        combined_features['neighbor_price_median'] = median_price
        combined_features['neighbor_price_std'] = price_std
        combined_features['neighbor_price_min'] = min_price
        combined_features['neighbor_price_max'] = max_price
        
        # Добавляем параметр months_on_sale (срок в продаже)
        combined_features['months_on_sale'] = 0
        
        # Шаг 4: Предсказываем цену с помощью взвешенного KNN 
        weights = 1 / (1 + neighbors['distance'])
        weights = weights / weights.sum()
        predicted_price = (neighbors[price_column] * weights).sum()
        prediction_method = 'weighted_knn'
        
        # Шаг 5: Находим похожие объекты
        similar_properties = find_similar_properties(predicted_price, _property_params, df, n_neighbors)
        final_neighbors = similar_properties if len(similar_properties) > 0 else neighbors
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Расчет метрик качества предсказания
        actual_prices = final_neighbors[price_column].values
        quality_metrics = None
        try:
            from scipy import stats
            
            neighbors_mae = mean_absolute_error(actual_prices, [predicted_price] * len(actual_prices))
            neighbors_rmse = math.sqrt(mean_squared_error(actual_prices, [predicted_price] * len(actual_prices)))
            
            def calculate_mape(y_true, y_pred):
                y_true, y_pred = np.array(y_true), np.array(y_pred)
                mask = y_true != 0
                return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
            
            neighbors_mape = calculate_mape(actual_prices, [predicted_price] * len(actual_prices))
            
            mean_price = np.mean(actual_prices)
            deviation_from_mean = abs(predicted_price - mean_price)
            deviation_from_mean_percent = (deviation_from_mean / mean_price * 100) if mean_price > 0 else 0
            
            median_price = np.median(actual_prices)
            deviation_from_median = abs(predicted_price - median_price)
            deviation_from_median_percent = (deviation_from_median / median_price * 100) if median_price > 0 else 0
            
            percentile = stats.percentileofscore(actual_prices, predicted_price)
            
            quality_metrics = {
                'neighbors_mae': neighbors_mae,
                'neighbors_rmse': neighbors_rmse,
                'neighbors_mape': neighbors_mape,
                'mean_price': mean_price,
                'median_price': median_price,
                'deviation_from_mean': deviation_from_mean,
                'deviation_from_mean_percent': deviation_from_mean_percent,
                'deviation_from_median': deviation_from_median,
                'deviation_from_median_percent': deviation_from_median_percent,
                'percentile': percentile
            }
        except Exception as e:
            st.warning(f"Не удалось рассчитать метрики качества: {e}")
        
        prediction_info = {
            'neighbors': final_neighbors.to_dict(orient='records'),
            'initial_neighbors': neighbors.to_dict(orient='records'),
            'predicted_price': predicted_price,
            'price_column': price_column,
            'weights': weights.tolist() if isinstance(weights, np.ndarray) else [1] * len(final_neighbors),
            'execution_time': execution_time,
            'prediction_method': prediction_method,
            'quality_metrics': quality_metrics,
            'neighbor_features': neighbor_features,
            'combined_features': combined_features
        }

        return predicted_price, prediction_info

    except Exception as e:
        st.error(f"Ошибка при предсказании цены: {str(e)}")
        st.error(traceback.format_exc())
        return 0.0, {}


def main():
    # Применяем кастомные CSS стили
    set_custom_css()
    st.title("RestatEval: Сервис предсказания стоимости жилья")
    
    # Проверка наличия файла данных
    if not os.path.exists(DATA_FILE):
        st.error(f"Файл данных {DATA_FILE} не найден.")
        uploaded_file = st.file_uploader("Загрузите CSV файл с данными о недвижимости", type="csv")
        
        if uploaded_file is not None:
            try:
                with open(DATA_FILE, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.success(f"Файл {DATA_FILE} успешно загружен!")
                st.experimental_rerun()
            except Exception as e:
                st.error(f"Ошибка при сохранении файла: {str(e)}")
        st.stop()
    
    # Загрузка пространственного индекса и модели
    if 'spatial_tree' not in st.session_state or 'xgboost_model' not in st.session_state:
        spatial_tree, spatial_df, xgboost_model = load_spatial_index_and_model()
        st.session_state['spatial_tree'] = spatial_tree
        st.session_state['spatial_df'] = spatial_df
        st.session_state['xgboost_model'] = xgboost_model
    
    # Загрузка датасета
    df = load_dataset(DATA_FILE)
    
    if df is None:
        st.error(f"Не удалось загрузить файл данных {DATA_FILE}. Убедитесь, что файл существует.")
        return
    
    # Создание индекса и модели при необходимости
    if (st.session_state['spatial_tree'] is None or 
        st.session_state['spatial_df'] is None or 
        st.session_state['xgboost_model'] is None):
        columns, df, spatial_tree, spatial_df, xgboost_model = create_housing_database()
        st.session_state['spatial_tree'] = spatial_tree
        st.session_state['spatial_df'] = spatial_df
        st.session_state['xgboost_model'] = xgboost_model
    
    # Основной интерфейс с двумя колонками (3:2 соотношение)
    col_map, col_params = st.columns([3, 2])
    
    # Колонка с картой
    with col_map:
        st.header("Интерактивная карта Москвы")
        st.info("Выберите местоположение объекта на карте, кликнув по ней и установив маркер.")
        
        # Создание и отображение карты
        map = create_interactive_map(df)
        map_data = st_folium(map, width=800, height=600)
        
        # Обработка выбранного местоположения
        selected_location = None
        if map_data and 'last_active_drawing' in map_data:
            if map_data['last_active_drawing'] and 'geometry' in map_data['last_active_drawing']:
                coords = map_data['last_active_drawing']['geometry']['coordinates']
                selected_location = {'lng': coords[0], 'lat': coords[1]}
                st.session_state['selected_location'] = selected_location
        
        # Отображение выбранных координат
        if 'selected_location' in st.session_state and st.session_state['selected_location']:
            loc = st.session_state['selected_location']
            st.success(f"Выбранные координаты: Широта {loc['lat']:.6f}, Долгота {loc['lng']:.6f}")
    
    # Колонка с параметрами
    with col_params:
        st.header("Параметры объекта")
        st.info("Введите параметры объекта для предсказания стоимости.")
        
        # Получение уникальных классов жилья
        class_values = ["Эконом", "Комфорт", "Комфорт+", "Бизнес", "Элит"]
        
        # Вкладки для разных методов расчета
        standard_tab, premium_tab = st.tabs(["Стандарт", "Премиум"])
        
        # Вкладка "Стандарт"
        with standard_tab:
            with st.form("standard_prediction_form"):
                st.write("")  # Отступ
                
                property_class = st.selectbox(
                    "Класс жилья", 
                    options=class_values, 
                    key="standard_class",
                    help="Выберите класс жилья из списка"
                )
                
                room_type = st.selectbox(
                    "Тип помещения",
                    options=["Квартира", "Апартаменты"],
                    key="standard_room_type",
                    help="Выберите тип помещения"
                )
                
                finish_type = st.selectbox(
                    "Отделка",
                    options=["Без отделки", "С отделкой"],
                    key="standard_finish",
                    help="Выберите тип отделки"
                )
                
                finish_value = 1.0 if finish_type == "С отделкой" else 0.0
                
                st.write("")  # Отступ
                
                col_year, col_month = st.columns(2)
                with col_year:
                    year = st.number_input(
                        "Год", 
                        min_value=2000, 
                        max_value=2030, 
                        value=2022, 
                        key="standard_year",
                        step=1,
                        format="%d"
                    )
                with col_month:
                    months = st.number_input(
                        "Месяц", 
                        min_value=1, 
                        max_value=12, 
                        value=6, 
                        key="standard_month",
                        step=1,
                        format="%d"
                    )
                
                st.write("")  # Отступ
                
                standard_submit = st.form_submit_button(
                    "Предсказать стоимость (5 соседей)",
                    use_container_width=True
                )
            
            # Обработка стандартного предсказания
            if standard_submit:
                if 'selected_location' not in st.session_state or not st.session_state['selected_location']:
                    st.warning("Пожалуйста, выберите местоположение на карте.")
                else:
                    loc = st.session_state['selected_location']
                    property_params = {
                        'lat': loc['lat'],
                        'lng': loc['lng'],
                        'Класс': property_class,
                        'Тип помещения': room_type,
                        'Отделка': finish_value,
                        'Дата ДДУ год': year,
                        'Дата ДДУ месяц': months,
                        'months_on_sale': 0
                    }
                    
                    with st.spinner("Выполняется предсказание..."):
                        result = predict_price_and_find_neighbors(
                            property_params, df, st.session_state['spatial_tree'], 
                            st.session_state['spatial_df'], st.session_state['xgboost_model'], n_neighbors=5
                        )
                    if result is not None:
                        predicted_price, prediction_info = result
                        st.session_state['prediction_info'] = prediction_info
                        st.session_state['predicted_price'] = predicted_price
                        
                        # Отображение таблицы с соседями под формой
                        st.subheader("Параметры похожих объектов")
                        neighbors_df = pd.DataFrame(prediction_info['neighbors'])
                        
                        if 'Отделка' in neighbors_df.columns:
                            neighbors_df['Отделка'] = neighbors_df['Отделка'].apply(
                                lambda x: "С отделкой" if x == 1.0 else "Без отделки"
                            )
                        
                        display_cols = ['ЖК', 'Класс', 'Цена со скидкой', 'Отделка', 'Дата ДДУ год', 'Дата ДДУ месяц', 'distance', 'months_on_sale', 'Номер квартиры']
                        display_cols = [col for col in display_cols if col in neighbors_df.columns]
                        
                        if 'distance' in neighbors_df.columns:
                            neighbors_df['distance'] = neighbors_df['distance'].round(2)
                        if 'Цена со скидкой' in neighbors_df.columns:
                            neighbors_df['Цена со скидкой'] = neighbors_df['Цена со скидкой'].astype(int)
                        
                        st.dataframe(
                            neighbors_df[display_cols],
                            column_config={
                                "distance": st.column_config.NumberColumn(
                                    "Расстояние (км)",
                                    format="%.2f км",
                                ),
                                "ЖК": st.column_config.TextColumn(
                                    "Жилой комплекс",
                                    width="medium"
                                ),
                                "months_on_sale": st.column_config.NumberColumn(
                                    "В продаже (мес.)",
                                    format="%d мес."
                                ),
                                "Номер квартиры": st.column_config.TextColumn(
                                    "Номер квартиры",
                                    width="medium"
                                ),
                                "Отделка": st.column_config.TextColumn(
                                    "Отделка",
                                    width="medium"
                                )
                            },
                            use_container_width=True,
                            height=min(400, 35 * len(neighbors_df) + 35)
                        )
                    else:
                        st.error("Не удалось выполнить предсказание. Проверьте входные данные.")
        
        # Вкладка "Премиум"
        with premium_tab:
            with st.form("premium_prediction_form"):
                st.write("")  # Отступ
                
                property_class_premium = st.selectbox(
                    "Класс жилья", 
                    options=class_values, 
                    key="premium_class",
                    help="Выберите класс жилья из списка"
                )
                
                room_type_premium = st.selectbox(
                    "Тип помещения",
                    options=["Квартира", "Апартаменты"],
                    key="premium_room_type",
                    help="Выберите тип помещения"
                )
                finish_type_premium = st.selectbox(
                    "Отделка",
                    options=["Без отделки", "С отделкой"],
                    key="premium_finish",
                    help="Выберите тип отделки"
                )
                
                finish_value_premium = 1.0 if finish_type_premium == "С отделкой" else 0.0
                
                floor = st.slider(
                    "Этаж",
                    min_value=1,
                    max_value=50,
                    value=10,
                    key="premium_floor",
                    help="Выберите этаж"
                )
                
                area = st.slider(
                    "Площадь (м²)",
                    min_value=30,
                    max_value=300,
                    value=70,
                    key="premium_area",
                    help="Выберите площадь помещения"
                )
                
                st.write("")  # Отступ
                
                col_year_p, col_month_p = st.columns(2)
                with col_year_p:
                    year_premium = st.number_input(
                        "Год", 
                        min_value=2000, 
                        max_value=2030, 
                        value=2022, 
                        key="premium_year",
                        step=1,
                        format="%d"
                    )
                with col_month_p:
                    months_premium = st.number_input(
                        "Месяц", 
                        min_value=1, 
                        max_value=12, 
                        value=6, 
                        key="premium_month",
                        step=1,
                        format="%d"
                    )
                
                st.write("")  # Отступ
                
                premium_submit = st.form_submit_button(
                    "Предсказать стоимость (15 соседей)",
                    use_container_width=True
                )
            
            # Обработка премиум предсказания
            if premium_submit:
                if 'selected_location' not in st.session_state or not st.session_state['selected_location']:
                    st.warning("Пожалуйста, выберите местоположение на карте.")
                else:
                    loc = st.session_state['selected_location']
                    property_params = {
                        'lat': loc['lat'],
                        'lng': loc['lng'],
                        'Класс': property_class_premium,
                        'Тип помещения': room_type_premium,
                        'Отделка': finish_value_premium,
                        'Этаж': floor,
                        'Площадь': area,
                        'Дата ДДУ год': year_premium,
                        'Дата ДДУ месяц': months_premium,
                        'months_on_sale': 0
                    }
                    
                    with st.spinner("Выполняется предсказание..."):
                        result = predict_price_and_find_neighbors(
                            property_params, df, st.session_state['spatial_tree'], 
                            st.session_state['spatial_df'], st.session_state['xgboost_model'], n_neighbors=15
                        )
                    if result is not None:
                        predicted_price, prediction_info = result
                        st.session_state['prediction_info'] = prediction_info
                        st.session_state['predicted_price'] = predicted_price
                        
                        st.subheader("Параметры похожих объектов")
                        neighbors_df = pd.DataFrame(prediction_info['neighbors'])
                        
                        if 'Отделка' in neighbors_df.columns:
                            neighbors_df['Отделка'] = neighbors_df['Отделка'].apply(
                                lambda x: "С отделкой" if x == 1.0 else "Без отделки"
                            )
                        
                        display_cols = ['ЖК', 'Класс', 'Цена со скидкой', 'Этаж', 'Площадь', 'Отделка', 'Дата ДДУ год', 'Дата ДДУ месяц', 'distance', 'months_on_sale', 'Номер квартиры']
                        display_cols = [col for col in display_cols if col in neighbors_df.columns]
                        
                        if 'distance' in neighbors_df.columns:
                            neighbors_df['distance'] = neighbors_df['distance'].round(2)
                        if 'Цена со скидкой' in neighbors_df.columns:
                            neighbors_df['Цена со скидкой'] = neighbors_df['Цена со скидкой'].astype(int)
                        
                        st.dataframe(
                            neighbors_df[display_cols],
                            column_config={
                                "distance": st.column_config.NumberColumn(
                                    "Расстояние (км)",
                                    format="%.2f км",
                                ),
                                "ЖК": st.column_config.TextColumn(
                                    "Жилой комплекс",
                                    width="medium"
                                ),
                                "months_on_sale": st.column_config.NumberColumn(
                                    "В продаже (мес.)",
                                    format="%d мес."
                                ),
                                "Номер квартиры": st.column_config.TextColumn(
                                    "Номер квартиры",
                                    width="medium"
                                ),
                                "Отделка": st.column_config.TextColumn(
                                    "Отделка",
                                    width="medium"
                                )
                            },
                            use_container_width=True,
                            height=min(400, 35 * len(neighbors_df) + 35)
                        )
                    else:
                        st.error("Не удалось выполнить предсказание. Проверьте входные данные.")
    
        # Отображение результатов под картой
    if 'prediction_info' in st.session_state and 'predicted_price' in st.session_state:
        st.divider()
        st.header("Результаты предсказания")
        st.success(f"Предсказанная стоимость: {int(st.session_state['predicted_price']):,} руб.")
        
        # Вкладки для графиков 
        tab1, tab2, tab3 = st.tabs([
            "Тенденция цены", 
            "Цена за м²", 
            "Влияние этажа"
        ])
        
        # Вкладка с графиком тенденции
        with tab1:
            st.subheader("Тенденция цены по похожим объектам")
            fig1 = plot_neighbors_price_trends(st.session_state['prediction_info'])
            if fig1:
                col1, col2, col3 = st.columns([1, 3, 1])
                with col2:
                    st.pyplot(fig1)
            else:
                st.info("Недостаточно данных для построения графика тенденций.")
        
        # Вкладка с анализом цены за квадратный метр
        with tab2:
            st.subheader("Анализ цены за квадратный метр")
            fig3 = plot_price_per_sqm(st.session_state['prediction_info'])
            if fig3:
                col1, col2, col3 = st.columns([1, 3, 1])
                with col2:
                    st.pyplot(fig3)
            else:
                st.info("Недостаточно данных для анализа цены за квадратный метр.")
        
        # Вкладка с анализом влияния этажа на цену
        with tab3:
            st.subheader("Влияние этажа на стоимость")
            fig4 = plot_price_by_floor(st.session_state['prediction_info'])
            if fig4:
                col1, col2, col3 = st.columns([1, 3, 1])
                with col2:
                    st.pyplot(fig4)
            else:
                st.info("Недостаточно данных для анализа влияния этажа на стоимость.")

if __name__ == "__main__":
    main()
