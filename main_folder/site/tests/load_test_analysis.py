# load_test_analysis.py
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import numpy as np

def analyze_locust_results(stats_file):
    """
    Анализ результатов нагрузочного тестирования Locust
    
    Args:
        stats_file (str): Путь к файлу с результатами тестирования
    """
    # Создаем директорию для результатов анализа
    results_dir = 'load_test_results'
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"Анализ файла: {stats_file}")
    
    try:
        # Загружаем данные
        with open(stats_file, 'r') as f:
            data = json.load(f)
        
        print("Данные успешно загружены")
        
        # Получаем список запросов
        requests = data['requests']
        print(f"Всего запросов: {len(requests)}")
        
        # Базовые метрики
        total_requests = len(requests)
        successful_requests = sum(1 for r in requests if r['success'])
        failed_requests = total_requests - successful_requests
        failure_rate = (failed_requests / total_requests * 100) if total_requests > 0 else 0
        
        # Время отклика
        response_times = [r['response_time'] for r in requests]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        
        # Сортируем для перцентилей
        response_times.sort()
        p50 = response_times[int(len(response_times) * 0.5)] if response_times else 0
        p95 = response_times[int(len(response_times) * 0.95)] if response_times else 0
        p99 = response_times[int(len(response_times) * 0.99)] if response_times else 0
        max_response_time = max(response_times) if response_times else 0
        
        # Расчет RPS
        timestamps = [datetime.fromisoformat(r['timestamp']) for r in requests]
        if len(timestamps) > 1:
            test_duration = (max(timestamps) - min(timestamps)).total_seconds()
            rps = total_requests / test_duration if test_duration > 0 else 0
        else:
            rps = 0
        
        # Создание визуализаций
        plt.style.use('seaborn-v0_8')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. График распределения времени отклика
        try:
            plt.figure(figsize=(12, 6))
            plt.hist(response_times, bins=50, alpha=0.7, color='blue')
            plt.axvline(avg_response_time, color='r', linestyle='dashed', label=f'Среднее ({avg_response_time:.1f}ms)')
            plt.axvline(p95, color='g', linestyle='dashed', label=f'95% ({p95:.1f}ms)')
            plt.title('Распределение времени отклика')
            plt.xlabel('Время отклика (мс)')
            plt.ylabel('Количество запросов')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(results_dir, f'response_time_distribution_{timestamp}.png'))
            plt.close()
            print("График распределения времени отклика создан")
        except Exception as e:
            print(f"Ошибка при создании графика распределения времени отклика: {e}")
        
        # 2. График успешных/неуспешных запросов
        try:
            plt.figure(figsize=(8, 8))
            plt.pie([successful_requests, failed_requests], 
                   labels=['Успешные', 'Неуспешные'],
                   autopct='%1.1f%%',
                   colors=['green', 'red'])
            plt.title('Распределение запросов')
            plt.savefig(os.path.join(results_dir, f'request_distribution_{timestamp}.png'))
            plt.close()
            print("График распределения запросов создан")
        except Exception as e:
            print(f"Ошибка при создании графика распределения запросов: {e}")
        
        # 3. График RPS во времени
        try:
            plt.figure(figsize=(12, 6))
            # Группируем запросы по секундам
            df = pd.DataFrame({'timestamp': timestamps})
            df['requests'] = 1
            df = df.set_index('timestamp')
            df = df.resample('1S').count()
            
            plt.plot(df.index, df['requests'], label='RPS')
            plt.title('Запросы в секунду (RPS)')
            plt.xlabel('Время')
            plt.ylabel('Запросов в секунду')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, f'rps_over_time_{timestamp}.png'))
            plt.close()
            print("График RPS создан")
        except Exception as e:
            print(f"Ошибка при создании графика RPS: {e}")
        
        # Создание отчета
        report = f"""
Отчет о нагрузочном тестировании
================================
Время проведения: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Общая статистика
---------------
Всего запросов: {total_requests}
Успешных запросов: {successful_requests}
Неуспешных запросов: {failed_requests}
Процент ошибок: {failure_rate:.2f}%
Запросов в секунду (RPS): {rps:.2f}

Время отклика (мс)
----------------
Среднее: {avg_response_time:.2f}
Медиана (P50): {p50:.2f}
95-й перцентиль (P95): {p95:.2f}
99-й перцентиль (P99): {p99:.2f}
Максимальное: {max_response_time:.2f}

Оценка производительности
-----------------------
1. Время отклика:
   - P95 < 1000 мс: {"ДА" if p95 < 1000 else "НЕТ"}
   - P99 < 2000 мс: {"ДА" if p99 < 2000 else "НЕТ"}

2. Стабильность:
   - Процент ошибок < 1%: {"ДА" if failure_rate < 1 else "НЕТ"}
   - RPS > 10: {"ДА" if rps > 10 else "НЕТ"}

Рекомендации
-----------
"""
        # Добавляем рекомендации на основе анализа
        if p95 >= 1000:
            report += """1. Оптимизация времени отклика:
   - Проверить настройки кэширования
   - Оптимизировать запросы к базе данных
   - Проанализировать узкие места в коде
"""
        
        if failure_rate >= 1:
            report += """2. Снижение количества ошибок:
   - Проверить обработку исключений
   - Увеличить таймауты
   - Проанализировать логи ошибок
"""
        
        if rps <= 10:
            report += """3. Повышение производительности:
   - Рассмотреть возможности масштабирования
   - Оптимизировать работу с базой данных
   - Проверить настройки сервера
"""
        
        # Сохранение отчета
        report_file = os.path.join(results_dir, f'load_test_report_{timestamp}.txt')
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\nОтчет сохранен в файл: {report_file}")
        print("\nКраткие результаты:")
        print(f"RPS: {rps:.2f}")
        print(f"Ошибки: {failure_rate:.2f}%")
        print(f"Среднее время отклика: {avg_response_time:.2f} мс")
        
        return {
            'total_requests': total_requests,
            'successful_requests': successful_requests,
            'failed_requests': failed_requests,
            'failure_rate': failure_rate,
            'avg_response_time': avg_response_time,
            'p95': p95,
            'p99': p99,
            'rps': rps
        }
        
    except Exception as e:
        print(f"Ошибка при анализе результатов: {e}")
        return None

def interpret_results(stats):
    """
    Интерпретация результатов нагрузочного тестирования
    
    Args:
        stats (dict): Статистика тестирования
    """
    if not stats:
        print("Нет данных для интерпретации")
        return
    
    print("\nИнтерпретация результатов:")
    print("==========================")
    
    # Оценка времени отклика
    print("\n1. Время отклика:")
    if stats['p95'] < 1000:
        print("✓ Отличное время отклика (P95 < 1000 мс)")
    elif stats['p95'] < 2000:
        print("⚠ Удовлетворительное время отклика (P95 < 2000 мс)")
    else:
        print("✗ Высокое время отклика (P95 >= 2000 мс)")
    
    # Оценка стабильности
    print("\n2. Стабильность системы:")
    if stats['failure_rate'] < 1:
        print("✓ Низкий процент ошибок (<1%)")
    elif stats['failure_rate'] < 5:
        print("⚠ Средний процент ошибок (<5%)")
    else:
        print("✗ Высокий процент ошибок (>=5%)")
    
    # Оценка производительности
    print("\n3. Производительность:")
    if stats['rps'] > 50:
        print("✓ Высокая производительность (>50 RPS)")
    elif stats['rps'] > 10:
        print("⚠ Средняя производительность (>10 RPS)")
    else:
        print("✗ Низкая производительность (<=10 RPS)")
    
    # Общие рекомендации
    print("\nРекомендации:")
    if stats['p95'] >= 1000:
        print("- Оптимизировать время отклика:")
        print("  * Улучшить кэширование")
        print("  * Оптимизировать запросы к БД")
    
    if stats['failure_rate'] >= 1:
        print("- Повысить стабильность:")
        print("  * Проверить обработку ошибок")
        print("  * Увеличить таймауты")
    
    if stats['rps'] <= 10:
        print("- Увеличить производительность:")
        print("  * Рассмотреть масштабирование")
        print("  * Оптимизировать код")

if __name__ == "__main__":
    # Пример использования
    stats_file = "locust_stats.json"
    
    if os.path.exists(stats_file):
        stats = analyze_locust_results(stats_file)
        if stats:
            interpret_results(stats)
    else:
        print(f"Файл {stats_file} не найден")
