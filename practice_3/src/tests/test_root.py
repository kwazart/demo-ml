from fastapi.testclient import TestClient
from practice_3.src.main import app

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hi there! It's a root page."}
