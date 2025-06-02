# locustfile.py
from locust import HttpUser, task, between
import json
from datetime import datetime

class WebsiteUser(HttpUser):
    host = "http://localhost:8000"
    wait_time = between(1, 2)
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stats = []
        self.csrf_token = None

    def on_start(self):
        """Получение CSRF токена при старте"""
        response = self.client.get("/")
        if 'csrftoken' in response.cookies:
            self.csrf_token = response.cookies['csrftoken']
            print(f"Получен CSRF токен: {self.csrf_token}")
        else:
            print("CSRF токен не найден в cookies!")
    
    @task
    def test_prediction(self):
        if not self.csrf_token:
            print("CSRF токен отсутствует, получаем новый")
            self.on_start()
            if not self.csrf_token:
                print("Не удалось получить CSRF токен")
                return

        # Заголовки с CSRF токеном
        headers = {
            'X-CSRFToken': self.csrf_token,
            'Content-Type': 'application/x-www-form-urlencoded',
            'Referer': f"{self.host}/"  # Django требует Referer для CSRF
        }
        
        # Данные запроса
        data = {
            "location_lat": "55.76",
            "location_lng": "37.64",
            "location_address": "Test Address",
            "floor": "10",
            "property_type": "Квартира",
            "finishing_type": "Без отделки",
            "property_class": "1",
            "purchase_month": "5",
            "purchase_year": "2025",
            "csrfmiddlewaretoken": self.csrf_token  # Добавляем токен в данные формы
        }

        # Отправка запроса с CSRF токеном
        with self.client.post(
            "/calculate-prediction/",  # Исправлен путь
            data=data,
            headers=headers,
            cookies={'csrftoken': self.csrf_token},
            catch_response=True
        ) as response:
            try:
                if response.status_code == 403:
                    print(f"Ошибка 403: {response.text}")
                    response.failure(f"CSRF ошибка: {response.text}")
                elif response.status_code != 200:
                    print(f"Ошибка {response.status_code}: {response.text}")
                    response.failure(f"Неожиданный статус: {response.status_code}")
                else:
                    response.success()
                    
                # Сохраняем статистику
                stat = {
                    'request_type': 'POST',
                    'name': '/calculate-prediction/',
                    'response_time': response.elapsed.total_seconds() * 1000,
                    'response_length': len(response.content) if response.content else 0,
                    'status_code': response.status_code,
                    'success': 200 <= response.status_code < 400,
                    'timestamp': datetime.now().isoformat()
                }
                self.stats.append(stat)
                
                # Сохраняем промежуточные результаты
                if len(self.stats) % 10 == 0:
                    self.save_stats()
                    
            except Exception as e:
                print(f"Ошибка при обработке ответа: {e}")
                response.failure(f"Ошибка: {str(e)}")

    def save_stats(self):
        """Сохранение статистики в файл"""
        try:
            with open('locust_stats.json', 'w') as f:
                json.dump({
                    'requests': self.stats,
                    'summary': {
                        'total_requests': len(self.stats),
                        'successful_requests': sum(1 for s in self.stats if s['success']),
                        'failed_requests': sum(1 for s in self.stats if not s['success']),
                        'average_response_time': sum(s['response_time'] for s in self.stats) / len(self.stats) if self.stats else 0
                    }
                }, f, indent=4)
        except Exception as e:
            print(f"Ошибка при сохранении статистики: {e}")

    def on_stop(self):
        """Сохранение финальной статистики при остановке"""
        self.save_stats()


