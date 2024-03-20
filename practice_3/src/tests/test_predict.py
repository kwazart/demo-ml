from fastapi.testclient import TestClient
from practice_3.src.main import app

client = TestClient(app)
url = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg"


def test_predict():
    response = client.post("/predict/", json={"url": url})
    text = response.text
    assert response.status_code == 200
    assert isinstance(text, str)
