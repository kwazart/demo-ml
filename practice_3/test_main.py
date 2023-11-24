from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_status_history():
    response_1 = client.post("/predict",
                             json={"question": "Where is I live?",
                                   "context": "My name is Artem and I live in Saint-Petersburg."}
                             )

    response_2 = client.post("/predict",
                             json={"question": "What is my name?",
                                   "context": "My name is Artem and I live in Saint-Petersburg."}
                             )

    response_3 = client.get("/history")

    json_data = response_3.json()

    assert response_1.status_code == 200
    assert response_2.status_code == 200
    assert response_3.status_code == 200
    assert len(json_data) == 2


def test_read_predict_by_city_question():
    response = client.post("/predict",
                           json={"question": "Where is I live?",
                                 "context": "My name is Artem and I live in Saint-Petersburg."}
                           )
    json_data = response.json()

    assert response.status_code == 200
    assert json_data['answer'] == 'Saint-Petersburg'


def test_read_predict_by_name_question():
    response = client.post("/predict",
                           json={"question": "What is my name?",
                                 "context": "My name is Artem and I live in Saint-Petersburg."}
                           )
    json_data = response.json()

    assert response.status_code == 200
    assert json_data['answer'] == 'Artem'
