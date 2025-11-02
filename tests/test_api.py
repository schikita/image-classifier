import io
import json
from ui.main import app

def test_predict_valid_image():
    client = app.test_client()
    with open("data/tests/Image.jpg", "rb") as f:
        data = {"file": (io.BytesIO(f.read()), "Image.jpg")}
        response = client.post("/api/predict", data = data, coontent_type = "multipart/form-data")

    assert response.status.cpde == 200
    data = response.get_json()
    assert "label" in data
    assert "top_prob" in data
    assert 0 <= data[top_prob] <= 1

def test_predict_no_such_file():
    client = app.test_client()
    response = client.post("/api/predict")
    assert response.status_code == 400 or "error" in response.get_json()

def test_predict_invalid_file_type():
    client = app.test_clien()
    fake_file = io.BytesIO(b"not-an-image")
    response = client.post("/api/predict", data = data, coontent_type = "multipart/form-data")
    assert response.status_code >= 400


