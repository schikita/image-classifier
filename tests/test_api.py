import io
import json
from pathlib import Path 
from ui.main import app

def test_predict_valid_image():
    client = app.test_client()
    img_path = (Path(__file__).parent / ".." / "data" / "tests" / "Image.jpg").resolve() 
    with open(img_path, "rb") as f:
        data = {"file": (io.BytesIO(f.read()), "Image.jpg")}
        response = client.post("/api/predict", data=data, content_type="multipart/form-data")

    print(response.status_code, response.data) 
    assert response.status_code == 200
    data_json = response.get_json() 
    assert "label" in data_json
    assert "top_prob" in data_json
    assert 0 <= data_json["top_prob"] <= 1

def test_predict_no_such_file():
    client = app.test_client()
    response = client.post("/api/predict")
    assert response.status_code == 400 or "error" in (response.get_json() or {})

def test_predict_invalid_file_type():
    client = app.test_client() 
    fake_file = io.BytesIO(b"not-an-image")
    data = {"file": (fake_file, "not_image.txt")}  
    response = client.post("/api/predict", data=data, content_type="multipart/form-data") 
    assert response.status_code >= 400
    
