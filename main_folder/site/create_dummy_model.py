import pickle
import numpy as np
import os

class DummyModel:
    def predict(self, X):
        """Имитация предсказания модели XGBoost"""
        # Просто возвращаем случайную цену в диапазоне от 5 до 15 миллионов рублей
        return np.array([np.random.uniform(5000000, 15000000)])

# Создаем экземпляр модели
model = DummyModel()

# Создаем директорию для модели, если она не существует
os.makedirs('analytics/models', exist_ok=True)

# Сохраняем модель в файл
with open('analytics/models/xgboost_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Dummy XGBoost model created successfully!")
