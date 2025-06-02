# tests/run_multiple_tests.py
# run_multiple_tests.py
import subprocess
import statistics
import json
from datetime import datetime
import os
import re
import sys
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import traceback

# Настройка логирования
print(f"Текущая директория: {os.getcwd()}")
print(f"Python версия: {sys.version}")
print("Matplotlib конфигурация:")
print(f"Backend: {plt.get_backend()}")

def extract_number(text):
    """Извлекает число из текста"""
    match = re.search(r'[-+]?\d*\.\d+|\d+', text)
    if match:
        return float(match.group())
    return None

def create_visualizations(cache_improvements, orm_improvements, results_dir, timestamp):
    """Создание визуализаций результатов тестирования"""
    try:
        # Проверяем входные данные
        if not cache_improvements or not orm_improvements:
            print("Нет данных для визуализации")
            return None

        print(f"Данные для визуализации:")
        print(f"cache_improvements: {cache_improvements}")
        print(f"orm_improvements: {orm_improvements}")

        # Проверяем и создаем директорию для графиков
        plots_dir = os.path.join(results_dir, 'plots')
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)
        
        print(f"Создание визуализаций в директории: {plots_dir}")

        # Проверяем права доступа
        try:
            test_file = os.path.join(plots_dir, 'test.txt')
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
        except Exception as e:
            print(f"Проблема с правами доступа к директории: {e}")
            return None

        # Настройка стиля
        plt.style.use('seaborn')
        sns.set_palette("husl")

        # 1. График распределения улучшений (Box Plot)
        try:
            plt.figure(figsize=(12, 6))
            data = [cache_improvements, orm_improvements]
            labels = ['Кэширование', 'ORM оптимизация']
            
            plt.boxplot(data, labels=labels)
            plt.title('Распределение улучшений производительности')
            plt.ylabel('Улучшение производительности (%)')
            plt.grid(True, linestyle='--', alpha=0.7)
            
            plt.axhline(y=40, color='r', linestyle='--', label='Цель кэширования (40%)')
            plt.axhline(y=30, color='g', linestyle='--', label='Цель ORM (30%)')
            plt.legend()
            
            boxplot_path = os.path.join(plots_dir, f'boxplot_{timestamp}.png')
            plt.savefig(boxplot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Сохранен график: {boxplot_path}")
            
            if not os.path.exists(boxplot_path):
                print(f"Файл {boxplot_path} не был создан!")
            else:
                print(f"Файл {boxplot_path} успешно создан")
                
        except Exception as e:
            print(f"Ошибка при создании boxplot: {e}")
            traceback.print_exc()

        # 2. Гистограммы распределения
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # Гистограмма кэширования
            ax1.hist(cache_improvements, bins=15, alpha=0.7, color='blue')
            ax1.axvline(np.mean(cache_improvements), color='r', linestyle='dashed', linewidth=2, label='Среднее')
            ax1.axvline(np.median(cache_improvements), color='g', linestyle='dashed', linewidth=2, label='Медиана')
            ax1.axvline(40, color='purple', linestyle='dashed', linewidth=2, label='Цель')
            ax1.set_title('Распределение улучшений кэширования')
            ax1.set_xlabel('Улучшение производительности (%)')
            ax1.set_ylabel('Частота')
            ax1.legend()
            
            # Гистограмма ORM
            ax2.hist(orm_improvements, bins=15, alpha=0.7, color='green')
            ax2.axvline(np.mean(orm_improvements), color='r', linestyle='dashed', linewidth=2, label='Среднее')
            ax2.axvline(np.median(orm_improvements), color='g', linestyle='dashed', linewidth=2, label='Медиана')
            ax2.axvline(30, color='purple', linestyle='dashed', linewidth=2, label='Цель')
            ax2.set_title('Распределение улучшений ORM')
            ax2.set_xlabel('Улучшение производительности (%)')
            ax2.set_ylabel('Частота')
            ax2.legend()
            
            plt.tight_layout()
            histograms_path = os.path.join(plots_dir, f'histograms_{timestamp}.png')
            plt.savefig(histograms_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Сохранен график: {histograms_path}")
            
            if not os.path.exists(histograms_path):
                print(f"Файл {histograms_path} не был создан!")
            else:
                print(f"Файл {histograms_path} успешно создан")
                
        except Exception as e:
            print(f"Ошибка при создании гистограмм: {e}")
            traceback.print_exc()

        # 3. График изменения улучшений по итерациям
        try:
            plt.figure(figsize=(12, 6))
            iterations = range(1, len(cache_improvements) + 1)
            
            plt.plot(iterations, cache_improvements, label='Кэширование', marker='o')
            plt.plot(iterations, orm_improvements, label='ORM оптимизация', marker='s')
            
            plt.axhline(y=40, color='r', linestyle='--', label='Цель кэширования (40%)')
            plt.axhline(y=30, color='g', linestyle='--', label='Цель ORM (30%)')
            
            plt.title('Изменение улучшений производительности по итерациям')
            plt.xlabel('Номер итерации')
            plt.ylabel('Улучшение производительности (%)')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            
            iterations_path = os.path.join(plots_dir, f'iterations_{timestamp}.png')
            plt.savefig(iterations_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Сохранен график: {iterations_path}")
            
            if not os.path.exists(iterations_path):
                print(f"Файл {iterations_path} не был создан!")
            else:
                print(f"Файл {iterations_path} успешно создан")
                
        except Exception as e:
            print(f"Ошибка при создании графика итераций: {e}")
            traceback.print_exc()

        # 4. Тепловая карта корреляции
        try:
            plt.figure(figsize=(8, 6))
            correlation_matrix = np.corrcoef([cache_improvements, orm_improvements])
            sns.heatmap(correlation_matrix, 
                        annot=True, 
                        cmap='coolwarm', 
                        xticklabels=['Кэширование', 'ORM'],
                        yticklabels=['Кэширование', 'ORM'])
            plt.title('Корреляция между улучшениями')
            
            correlation_path = os.path.join(plots_dir, f'correlation_{timestamp}.png')
            plt.savefig(correlation_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Сохранен график: {correlation_path}")
            
            if not os.path.exists(correlation_path):
                print(f"Файл {correlation_path} не был создан!")
            else:
                print(f"Файл {correlation_path} успешно создан")
                
        except Exception as e:
            print(f"Ошибка при создании тепловой карты: {e}")
            traceback.print_exc()

        return plots_dir

    except Exception as e:
        print(f"Общая ошибка при создании визуализаций: {e}")
        traceback.print_exc()
        return None

def run_tests(num_runs=30):
    """
    Выполняет тесты производительности и создает отчет с визуализациями
    
    Args:
        num_runs (int): Количество запусков тестов
    """
    cache_improvements = []
    orm_improvements = []
    
    print(f"Запуск {num_runs} итераций тестирования...")
    
    # Создаем директорию для результатов
    results_dir = 'test_results'
    os.makedirs(results_dir, exist_ok=True)
    
    # Основной цикл тестирования
    for i in range(num_runs):
        print(f"\nЗапуск теста {i+1}/{num_runs}")
        
        try:
            # Запуск тестов Django
            result = subprocess.run(
                [sys.executable, 'manage.py', 'test', 'core.tests', '-v', '2'],
                capture_output=True,
                text=True,
                encoding='cp1251'
            )
            
            # Сохраняем вывод теста
            test_output_file = os.path.join(results_dir, f'test_output_{i+1}.txt')
            with open(test_output_file, 'w', encoding='cp1251') as f:
                if result.stdout:
                    f.write("STDOUT:\n")
                    f.write(result.stdout)
                if result.stderr:
                    f.write("\nSTDERR:\n")
                    f.write(result.stderr)
            
            # Обработка вывода тестов
            if result.stdout:
                lines = result.stdout.split('\n')
                for line in lines:
                    # Поиск результатов кэширования
                    if "Кэширование улучшило производительность на" in line:
                        try:
                            improvement = float(line.split("на")[1].strip().rstrip('%'))
                            cache_improvements.append(improvement)
                            print(f"Найдено улучшение кэширования: {improvement}%")
                        except Exception as e:
                            print(f"Ошибка при парсинге результата кэширования: {e}")
                            print(f"Проблемная строка: {line}")
                    
                    # Поиск результатов ORM
                    if "Оптимизация ORM улучшила производительность на" in line:
                        try:
                            improvement = float(line.split("на")[1].strip().rstrip('%'))
                            orm_improvements.append(improvement)
                            print(f"Найдено улучшение ORM: {improvement}%")
                        except Exception as e:
                            print(f"Ошибка при парсинге результата ORM: {e}")
                            print(f"Проблемная строка: {line}")

            # Выводим промежуточные результаты
            print(f"Текущее количество результатов:")
            print(f"Кэширование: {len(cache_improvements)}")
            print(f"ORM: {len(orm_improvements)}")

        except Exception as e:
            print(f"Ошибка при выполнении теста {i+1}: {e}")
            continue

    # Проверка наличия результатов
    if not cache_improvements or not orm_improvements:
        print("\nПРЕДУПРЕЖДЕНИЕ: Не удалось получить результаты тестов!")
        print(f"Количество результатов кэширования: {len(cache_improvements)}")
        print(f"Количество результатов ORM: {len(orm_improvements)}")
        return

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

    # Вычисление статистик
    cache_stats = calculate_stats(cache_improvements)
    orm_stats = calculate_stats(orm_improvements)

    # Генерация временной метки
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Создание визуализаций
    plots_dir = None
    try:
        print("\nНачало создания визуализаций...")
        plots_dir = create_visualizations(cache_improvements, orm_improvements, results_dir, timestamp)
        if plots_dir and os.path.exists(plots_dir):
            print(f"\nГрафики успешно сохранены в папке: {plots_dir}")
            # Проверяем наличие файлов
            expected_files = [
                f'boxplot_{timestamp}.png',
                f'histograms_{timestamp}.png',
                f'iterations_{timestamp}.png',
                f'correlation_{timestamp}.png'
            ]
            for file in expected_files:
                full_path = os.path.join(plots_dir, file)
                if os.path.exists(full_path):
                    print(f"Файл {file} создан успешно")
                else:
                    print(f"Файл {file} не найден!")
        else:
            print("\nНе удалось создать графики")
            print(f"plots_dir: {plots_dir}")
    except Exception as e:
        print(f"Ошибка при создании визуализаций: {e}")
        traceback.print_exc()

    # Формирование отчета
    report = f"""Результаты тестирования ({num_runs} запусков)
Время проведения: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Гипотеза 1.1 (Кэширование):
- Среднее улучшение: {cache_stats['mean']:.2f}%
- Медиана улучшения: {cache_stats['median']:.2f}%
- Стандартное отклонение: {cache_stats['std_dev']:.2f}%
- Минимальное улучшение: {cache_stats['min']:.2f}%
- Максимальное улучшение: {cache_stats['max']:.2f}%

Гипотеза 1.2 (Оптимизация ORM):
- Среднее улучшение: {orm_stats['mean']:.2f}%
- Медиана улучшения: {orm_stats['median']:.2f}%
- Стандартное отклонение: {orm_stats['std_dev']:.2f}%
- Минимальное улучшение: {orm_stats['min']:.2f}%
- Максимальное улучшение: {orm_stats['max']:.2f}%

Вывод:
1. Гипотеза 1.1 {"ПОДТВЕРЖДЕНА" if cache_stats['mean'] >= 40 else "НЕ ПОДТВЕРЖДЕНА"}
   (целевое улучшение: 40%, достигнуто: {cache_stats['mean']:.2f}%)
   
2. Гипотеза 1.2 {"ПОДТВЕРЖДЕНА" if orm_stats['mean'] >= 30 else "НЕ ПОДТВЕРЖДЕНА"}
   (целевое улучшение: 30%, достигнуто: {orm_stats['mean']:.2f}%)

Детальные результаты:
Кэширование: {cache_improvements}
ORM: {orm_improvements}

Визуализации сохранены в папке: {plots_dir or 'не созданы'}
1. boxplot_{timestamp}.png - Распределение улучшений производительности
2. histograms_{timestamp}.png - Гистограммы распределения улучшений
3. iterations_{timestamp}.png - График изменения улучшений по итерациям
4. correlation_{timestamp}.png - Тепловая карта корреляции
"""

    # Сохранение отчета
    report_file = os.path.join(results_dir, f'test_report_{timestamp}.txt')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)

    # Сохранение сырых данных
    raw_data = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'num_runs': num_runs,
        'cache_improvements': cache_improvements,
        'orm_improvements': orm_improvements,
        'cache_stats': cache_stats,
        'orm_stats': orm_stats
    }
    
    json_file = os.path.join(results_dir, f'raw_data_{timestamp}.json')
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(raw_data, f, indent=4)

    # Вывод результатов
    print("\nРезультаты тестирования:")
    print(report)
    print(f"\nПодробный отчет сохранен в файл: {report_file}")
    print(f"Сырые данные сохранены в файл: {json_file}")

if __name__ == "__main__":
    run_tests(30)
