import os
import sys

from fastapi.testclient import TestClient

# так как запуск происходит из папки practice_5-domchenko, в sys.path добавляем её для импорта main
sys.path.append(os.path.join(os.getcwd()))
from main import app

client = TestClient(app)


def test_healthcheck_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {'message': 'Hello World!!!'}


def test_predict_car():
    response = client.post("/predict/",
                           json={
                               "url": "https://parkingcars.ru/wp-content/uploads/2021/02/stoyanka-1024x683.jpg",
                               "targets": "car"
                           })
    json_data = response.json()

    assert response.status_code == 200
    assert len(json_data) == 14
    for res in json_data:
        assert res["label"] in ("car", "minivan")  # minivan нашло автоопределение


def test_predict_pineapple():
    response = client.post("/predict/",
                           json={
                               "url": "https://storage.yandexcloud.net/mfi/1242/products/main/3474.jpg",
                               "targets": "pineapple"
                           })
    json_data = response.json()

    assert response.status_code == 200
    assert len(json_data) == 2
    for res in json_data:
        assert res["label"] == "pineapple"
