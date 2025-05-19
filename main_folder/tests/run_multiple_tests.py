# tests/run_multiple_tests.py
import subprocess
import statistics
import json
from datetime import datetime
import os

def run_tests(num_runs=30):
    cache_improvements = []
    orm_improvements = []
    
    print(f"Запуск {num_runs} итераций тестирования...")
    
    for i in range(num_runs):
        print(f"\nЗапуск теста {i+1}/{num_runs}")
        
        # Запуск тестов и получение вывода
        result = subprocess.run(['python', 'manage.py', 'test', 'core', '-v', '2'], 
                              capture_output=True, text=True)
        
        # Поиск результатов в выводе
        for line in result.stdout.split('\n'):
            if "Кэширование улучшило производительность на" in line:
                try:
                    improvement = float(line.split("на")[-1].replace('%', '').strip())
                    cache_improvements.append(improvement)
                    print(f"Улучшение кэширования: {improvement}%")
                except:
                    pass
                    
            if "Оптимизация ORM улучшила производительность на" in line:
                try:
                    improvement = float(line.split("на")[-1].replace('%', '').strip())
                    orm_improvements.append(improvement)
                    print(f"Улучшение ORM: {improvement}%")
                except:
                    pass

    # Расчет статистики
    stats = {
        'cache': {
            'mean': statistics.mean(cache_improvements) if cache_improvements else 0,
            'median': statistics.median(cache_improvements) if cache_improvements else 0,
            'std_dev': statistics.stdev(cache_improvements) if len(cache_improvements) > 1 else 0,
            'min': min(cache_improvements) if cache_improvements else 0,
            'max': max(cache_improvements) if cache_improvements else 0,
            'all_values': cache_improvements
        },
        'orm': {
            'mean': statistics.mean(orm_improvements) if orm_improvements else 0,
            'median': statistics.median(orm_improvements) if orm_improvements else 0,
            'std_dev': statistics.stdev(orm_improvements) if len(orm_improvements) > 1 else 0,
            'min': min(orm_improvements) if orm_improvements else 0,
            'max': max(orm_improvements) if orm_improvements else 0,
            'all_values': orm_improvements
        }
    }

    # Создание отчета
    report = f"""
Результаты тестирования ({num_runs} запусков)
Время проведения: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Гипотеза 1.1 (Кэширование):
- Среднее улучшение: {stats['cache']['mean']:.2f}%
- Медиана улучшения: {stats['cache']['median']:.2f}%
- Стандартное отклонение: {stats['cache']['std_dev']:.2f}%
- Минимальное улучшение: {stats['cache']['min']:.2f}%
- Максимальное улучшение: {stats['cache']['max']:.2f}%

Гипотеза 1.2 (Оптимизация ORM):
- Среднее улучшение: {stats['orm']['mean']:.2f}%
- Медиана улучшения: {stats['orm']['median']:.2f}%
- Стандартное отклонение: {stats['orm']['std_dev']:.2f}%
- Минимальное улучшение: {stats['orm']['min']:.2f}%
- Максимальное улучшение: {stats['orm']['max']:.2f}%

Вывод:
1. Гипотеза 1.1 {"ПОДТВЕРЖДЕНА" if stats['cache']['mean'] >= 40 else "НЕ ПОДТВЕРЖДЕНА"}
   (целевое улучшение: 40%, достигнуто: {stats['cache']['mean']:.2f}%)
   
2. Гипотеза 1.2 {"ПОДТВЕРЖДЕНА" if stats['orm']['mean'] >= 30 else "НЕ ПОДТВЕРЖДЕНА"}
   (целевое улучшение: 30%, достигнуто: {stats['orm']['mean']:.2f}%)
"""

    # Сохранение результатов
    os.makedirs('test_results', exist_ok=True)
    
    # Сохранение подробного отчета
    with open(f'test_results/test_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt', 'w') as f:
        f.write(report)
    
    # Сохранение сырых данных для дальнейшего анализа
    with open(f'test_results/test_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json', 'w') as f:
        json.dump(stats, f, indent=4)

    print("\nОтчет сохранен в папке test_results")
    print(report)

if __name__ == "__main__":
    run_tests(30)  # Запуск 30 итераций тестов
