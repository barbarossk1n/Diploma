# run_multiple_tests.py
import subprocess
import statistics
import json
from datetime import datetime
import os
import re
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import traceback
import streamlit as st
import pandas as pd
from io import BytesIO

# Настройка стилей графиков
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def extract_number(text):
    """Извлекает число из текста"""
    match = re.search(r'[-+]?\d*\.\d+|\d+', text)
    if match:
        return float(match.group())
    return None

def create_visualizations(cache_improvements, orm_improvements):
    """Создание визуализаций и вывод в Streamlit"""
    try:
        # Установка современного стиля
        plt.style.use('seaborn-v0_8')  # Или 'ggplot', 'bmh', 'dark_background' и др.
        sns.set_theme(style="whitegrid")  # Альтернатива через seaborn
        
        # Проверка входных данных
        if not cache_improvements or not orm_improvements:
            st.error("Нет данных для визуализации")
            return

        # Создаем вкладки для разных графиков
        tab1, tab2, tab3, tab4 = st.tabs([
            "Распределение улучшений", 
            "Гистограммы",
            "Динамика по итерациям",
            "Корреляция"
        ])

        # 1. График распределения улучшений (Box Plot)
        with tab1:
            st.subheader("Распределение улучшений производительности")
            fig, ax = plt.subplots(figsize=(10, 5))
            data = [cache_improvements, orm_improvements]
            labels = ['Кэширование', 'ORM оптимизация']
            
            ax.boxplot(data, labels=labels)
            ax.set_ylabel('Улучшение производительности (%)')
            ax.grid(True, linestyle='--', alpha=0.7)
            
            ax.axhline(y=40, color='r', linestyle='--', label='Цель кэширования (40%)')
            ax.axhline(y=30, color='g', linestyle='--', label='Цель ORM (30%)')
            ax.legend()
            
            st.pyplot(fig)
            plt.close()

        # 2. Гистограммы распределения
        with tab2:
            st.subheader("Гистограммы распределения улучшений")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # Гистограмма кэширования
            ax1.hist(cache_improvements, bins=15, alpha=0.7, color='blue')
            ax1.axvline(np.mean(cache_improvements), color='r', linestyle='dashed', linewidth=2, label='Среднее')
            ax1.axvline(np.median(cache_improvements), color='g', linestyle='dashed', linewidth=2, label='Медиана')
            ax1.axvline(40, color='purple', linestyle='dashed', linewidth=2, label='Цель')
            ax1.set_title('Кэширование')
            ax1.set_xlabel('Улучшение (%)')
            ax1.set_ylabel('Частота')
            ax1.legend()
            
            # Гистограмма ORM
            ax2.hist(orm_improvements, bins=15, alpha=0.7, color='green')
            ax2.axvline(np.mean(orm_improvements), color='r', linestyle='dashed', linewidth=2, label='Среднее')
            ax2.axvline(np.median(orm_improvements), color='g', linestyle='dashed', linewidth=2, label='Медиана')
            ax2.axvline(30, color='purple', linestyle='dashed', linewidth=2, label='Цель')
            ax2.set_title('ORM оптимизация')
            ax2.set_xlabel('Улучшение (%)')
            ax2.set_ylabel('Частота')
            ax2.legend()
            
            st.pyplot(fig)
            plt.close()

        # 3. График изменения улучшений по итерациям
        with tab3:
            st.subheader("Динамика улучшений по итерациям")
            fig, ax = plt.subplots(figsize=(12, 6))
            iterations = range(1, len(cache_improvements) + 1)
            
            ax.plot(iterations, cache_improvements, label='Кэширование', marker='o')
            ax.plot(iterations, orm_improvements, label='ORM оптимизация', marker='s')
            
            ax.axhline(y=40, color='r', linestyle='--', label='Цель кэширования (40%)')
            ax.axhline(y=30, color='g', linestyle='--', label='Цель ORM (30%)')
            
            ax.set_title('Изменение улучшений по итерациям')
            ax.set_xlabel('Номер итерации')
            ax.set_ylabel('Улучшение (%)')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.7)
            
            st.pyplot(fig)
            plt.close()

        # 4. Тепловая карта корреляции
        with tab4:
            st.subheader("Корреляция между улучшениями")
            fig, ax = plt.subplots(figsize=(8, 6))
            correlation_matrix = np.corrcoef([cache_improvements, orm_improvements])
            sns.heatmap(correlation_matrix, 
                      annot=True, 
                      cmap='coolwarm', 
                      xticklabels=['Кэширование', 'ORM'],
                      yticklabels=['Кэширование', 'ORM'],
                      ax=ax)
            ax.set_title('Корреляция между улучшениями')
            
            st.pyplot(fig)
            plt.close()

    except Exception as e:
        st.error(f"Ошибка при создании визуализаций: {str(e)}")
        traceback.print_exc()

def display_results(cache_improvements, orm_improvements, num_runs):
    """Отображение результатов в Streamlit"""
    # Расчет статистики
    def calculate_stats(improvements):
        if not improvements:
            return {
                'mean': 0.0,
                'median': 0.0,
                'std_dev': 0.0,
                'min': 0.0,
                'max': 0.0
            }
        return {
            'mean': statistics.mean(improvements),
            'median': statistics.median(improvements),
            'std_dev': statistics.stdev(improvements) if len(improvements) > 1 else 0.0,
            'min': min(improvements),
            'max': max(improvements)
        }

    cache_stats = calculate_stats(cache_improvements)
    orm_stats = calculate_stats(orm_improvements)

    # Основные метрики
    st.title('Результаты тестирования производительности')
    st.write(f"Количество запусков тестов: {num_runs}")
    st.write(f"Время проведения: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Кэширование")
        st.metric("Среднее улучшение", f"{cache_stats['mean']:.2f}%", 
                 delta=f"Цель: 40% ({cache_stats['mean']-40:.2f}%)", delta_color="normal")
        st.metric("Медиана улучшения", f"{cache_stats['median']:.2f}%")
        st.metric("Стандартное отклонение", f"{cache_stats['std_dev']:.2f}%")
        
        # Вывод гипотезы
        if cache_stats['mean'] >= 40:
            st.success("✅ Гипотеза 1.1 ПОДТВЕРЖДЕНА (целевое улучшение: 40%)")
        else:
            st.error("❌ Гипотеза 1.1 НЕ ПОДТВЕРЖДЕНА (целевое улучшение: 40%)")
    
    with col2:
        st.subheader("ORM оптимизация")
        st.metric("Среднее улучшение", f"{orm_stats['mean']:.2f}%", 
                 delta=f"Цель: 30% ({orm_stats['mean']-30:.2f}%)", delta_color="normal")
        st.metric("Медиана улучшения", f"{orm_stats['median']:.2f}%")
        st.metric("Стандартное отклонение", f"{orm_stats['std_dev']:.2f}%")
        
        # Вывод гипотезы
        if orm_stats['mean'] >= 30:
            st.success("✅ Гипотеза 1.2 ПОДТВЕРЖДЕНА (целевое улучшение: 30%)")
        else:
            st.error("❌ Гипотеза 1.2 НЕ ПОДТВЕРЖДЕНА (целевое улучшение: 30%)")
    
    st.markdown("---")
    
    # Визуализации
    create_visualizations(cache_improvements, orm_improvements)
    
    # Сырые данные
    with st.expander("Показать сырые данные"):
        st.subheader("Кэширование")
        st.dataframe(pd.DataFrame(cache_improvements, columns=['Улучшение, %']).T)
        
        st.subheader("ORM оптимизация")
        st.dataframe(pd.DataFrame(orm_improvements, columns=['Улучшение, %']).T)

def run_tests(num_runs=30):
    """Запуск тестов и обработка результатов"""
    cache_improvements = []
    orm_improvements = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(num_runs):
        status_text.text(f"Запуск теста {i+1}/{num_runs}...")
        progress_bar.progress((i + 1) / num_runs)
        
        try:
            result = subprocess.run(
                [sys.executable, 'manage.py', 'test', 'core.tests', '-v', '2'],
                capture_output=True,
                text=True,
                encoding='cp1251'
            )
            
            if result.stdout:
                for line in result.stdout.split('\n'):
                    if "Кэширование улучшило производительность на" in line:
                        try:
                            improvement = float(line.split("на")[1].strip().rstrip('%'))
                            cache_improvements.append(improvement)
                        except Exception as e:
                            print(f"Ошибка парсинга: {e}")
                    
                    if "Оптимизация ORM улучшила производительность на" in line:
                        try:
                            improvement = float(line.split("на")[1].strip().rstrip('%'))
                            orm_improvements.append(improvement)
                        except Exception as e:
                            print(f"Ошибка парсинга: {e}")

        except Exception as e:
            print(f"Ошибка при выполнении теста {i+1}: {e}")
            continue

    # Проверка результатов
    if not cache_improvements or not orm_improvements:
        st.error("Не удалось получить результаты тестов!")
        return None, None
    
    return cache_improvements, orm_improvements

def main():
    st.set_page_config(page_title="Тестирование производительности", layout="wide")
    
    st.sidebar.title("Настройки тестирования")
    num_runs = st.sidebar.number_input("Количество запусков", min_value=1, max_value=100, value=30)
    
    if st.sidebar.button("Запустить тестирование"):
        with st.spinner("Выполнение тестов..."):
            cache_data, orm_data = run_tests(num_runs)
        
        if cache_data and orm_data:
            display_results(cache_data, orm_data, num_runs)

if __name__ == "__main__":
    main()