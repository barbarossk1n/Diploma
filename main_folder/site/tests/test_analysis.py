 # tests/test_analysis.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.tests.test_caching import test_with_cache, test_without_cache
from core.tests.test_orm import test_optimized_query, test_unoptimized_query

def analyze_results():
    print("=== Результаты тестирования кэширования ===")
    no_cache_time = test_without_cache()
    with_cache_time = test_with_cache()
    cache_improvement = ((no_cache_time - with_cache_time) / no_cache_time) * 100
    
    print(f"Время без кэша: {no_cache_time:.2f} сек")
    print(f"Время с кэшем: {with_cache_time:.2f} сек")
    print(f"Улучшение производительности: {cache_improvement:.1f}%")
    
    print("\n=== Результаты оптимизации ORM ===")
    unoptimized_time = test_unoptimized_query()
    optimized_time = test_optimized_query()
    orm_improvement = ((unoptimized_time - optimized_time) / unoptimized_time) * 100
    
    print(f"Время без оптимизации: {unoptimized_time:.2f} сек")
    print(f"Время с оптимизацией: {optimized_time:.2f} сек")
    print(f"Улучшение производительности: {orm_improvement:.1f}%")

if __name__ == "__main__":
    analyze_results()

