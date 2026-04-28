from fastapi.testclient import TestClient
from src.backend.main import app
from unittest.mock import patch
import pytest

client = TestClient(app)

def test_predict_endpoint():
    with patch("src.backend.main.get_predictions") as mock_pred:
        mock_pred.return_value = [{"ds": "2023-01-01", "yhat": 10}]
        response = client.get("/predict?periods=10")
        assert response.status_code == 200
        assert response.json() == [{"ds": "2023-01-01", "yhat": 10}]
        mock_pred.assert_called_with(10)

def test_predict_risk_endpoint():
    with patch("src.backend.main.get_predictive_risk") as mock_risk:
        mock_risk.return_value = {"risk": "LOW", "peak_prediction": 5}
        response = client.get("/predict-risk?periods=15")
        assert response.status_code == 200
        assert response.json()["risk"] == "LOW"
        mock_risk.assert_called_with(15)

def test_train_endpoint():
    with patch("src.backend.main.train_system") as mock_train:
        mock_train.return_value = {"status": "trained"}
        response = client.post("/train")
        assert response.status_code == 200
        assert response.json()["status"] == "trained"

def test_live_density_endpoint():
    with patch("src.backend.main.process_image") as mock_proc, \
         patch("src.backend.main.generate_alert") as mock_alert:
        
        mock_proc.return_value = {"count": 10}
        mock_alert.return_value = {"level": "LOW", "alert": False}
        
        # Create a dummy file
        files = {"file": ("test.jpg", b"fake image content", "image/jpeg")}
        response = client.post("/live-density?zone_name=Test", files=files)
        
        assert response.status_code == 200
        assert response.json()["count"] == 10
        assert response.json()["risk"]["level"] == "LOW"

def test_predict_zone_endpoint():
    with patch("src.backend.main.get_predictive_risk") as mock_risk:
        mock_risk.return_value = {"risk": "LOW"}
        response = client.post("/predict-zone", json={"periods": 10, "history_counts": [1, 2, 3]})
        assert response.status_code == 200
        assert response.json()["risk"] == "LOW"
        mock_risk.assert_called_with(10, [1, 2, 3])

def test_test_email_alert_endpoint():
    with patch("src.backend.main.generate_alert") as mock_alert:
        mock_alert.return_value = {"level": "HIGH ALERT", "alert": True}
        response = client.get("/test-email-alert?count=35")
        assert response.status_code == 200
        assert response.json()["level"] == "HIGH ALERT"
        mock_alert.assert_called_with(35, "Test Zone", 25)
