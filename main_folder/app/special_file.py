import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import scipy.sparse as sp
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import time

file_path = 'merge_TOTAL.csv'

# Функция для загрузки данных по частям
def load_data_in_chunks(file_path, chunk_size=10000):
    # Сначала определим количество строк и колонок для предварительного создания массивов
    total_rows = 0
    for chunk in pd.read_csv(file_path, chunksize=1):
        # Получаем имена колонок из первого чанка
        columns = chunk.columns
        break
    
    # Подсчитываем общее количество строк
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        total_rows += len(chunk)
    
    print(f"Всего строк в файле: {total_rows}")
    
    # Загружаем данные по частям и преобразуем в разреженные матрицы
    chunks = []
    y_values = []
    
    for i, chunk in enumerate(pd.read_csv(file_path, chunksize=chunk_size)):
        print(f"Обработка чанка {i+1}...")
        
        # Предобработка
        if 'ЖК рус' in chunk.columns:
            chunk = chunk.drop(columns=['ЖК рус'])
        chunk = chunk.fillna(0).astype(float)
        
        # Разделение на признаки и целевую переменную
        if 'Цена кв.м' in chunk.columns:
            y_chunk = chunk['Цена кв.м'].values
            X_chunk = chunk.drop('Цена кв.м', axis=1)
        else:
            raise ValueError("В данных отсутствует столбец 'Цена кв.м'")
        
        # Преобразование в разреженную матрицу
        chunks.append(sp.csr_matrix(X_chunk.values))
        y_values.extend(y_chunk)
    
    # Объединяем все чанки
    X_sparse = sp.vstack(chunks)
    y = np.array(y_values)
    
    return X_sparse, y, list(X_chunk.columns)


# Загрузка данных
print('Загрузка данных...')
X_sparse, y, feature_names = load_data_in_chunks(file_path, chunk_size=50000)
print(f'Форма данных: X: {X_sparse.shape}, y: {y.shape}')

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X_sparse, y, test_size=0.2, random_state=42)

print(f'Размер обучающей выборки: {X_train.shape}')
print(f'Размер тестовой выборки: {X_test.shape}')

# Преобразование разреженных матриц в формат DMatrix для XGBoost
print('Преобразование данных в DMatrix...')
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Определение параметров модели
params = {
    'objective': 'reg:squarederror',  # задача регрессии
    'eval_metric': 'rmse',            # метрика для оценки
    'eta': 0.1,                       # скорость обучения
    'max_depth': 6,                   # максимальная глубина дерева
    'subsample': 0.8,                 # доля объектов для обучения
    'colsample_bytree': 0.8,          # доля признаков для обучения
    'min_child_weight': 1,            # минимальное количество объектов в листе
    'nthread': -1                     # использовать все доступные ядра процессора
}

# Задаем список данных для оценки
eval_list = [(dtrain, 'train'), (dtest, 'eval')]

# Обучение модели с выводом метрик
print('Начало обучения XGBoost модели...')
start_time = time.time()
model = xgb.train(
    params,
    dtrain,
    num_boost_round=100,              # количество итераций
    evals=eval_list,
    early_stopping_rounds=10,         # ранняя остановка, если нет улучшения
    verbose_eval=10                   # выводить метрики каждые 10 итераций
)
training_time = time.time() - start_time
print(f'Обучение завершено за {training_time:.2f} секунд')

# Предсказания на тестовой выборке
print('Оценка модели на тестовой выборке...')
y_pred = model.predict(dtest)

# Расчет метрик
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f'Средняя абсолютная ошибка (MAE): {mae:.2f}')
print(f'Корень из среднеквадратичной ошибки (RMSE): {rmse:.2f}')
print(f'Коэффициент детерминации (R²): {r2:.4f}')

# Сохранение модели
print('Сохранение модели в файл...')
with open('xgboost_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print('Модель успешно сохранена в файл xgboost_model.pkl')

# Вывод важности признаков
print('Топ-10 важных признаков:')
importance = model.get_score(importance_type='gain')
importance = {k: v for k, v in sorted(importance.items(), key=lambda item: item[1], reverse=True)}

# Преобразуем номера признаков в их имена
feature_importance = {}
for k, v in importance.items():
    feature_idx = int(k.replace('f', ''))
    if feature_idx < len(feature_names):
        feature_importance[feature_names[feature_idx]] = v
    else:
        feature_importance[k] = v

# Вывод топ-10 важных признаков
for i, (feature, score) in enumerate(list(feature_importance.items())[:10]):
    print(f"{i+1}. {feature}: {score:.4f}")
