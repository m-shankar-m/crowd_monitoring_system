import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from src.backend.main import app

client = TestClient(app)

def test_read_main():
    # Test a simple get if applicable, but main.py doesn't have a root GET
    # Let's test the predict-risk endpoint
    with patch("src.backend.main.get_predictive_risk") as mock_risk:
        mock_risk.return_value = {"peak": 10, "risk": "low"}
        response = client.get("/predict-risk?periods=10")
        assert response.status_code == 200
        assert response.json() == {"peak": 10, "risk": "low"}

def test_test_email_alert_endpoint():
    with patch("src.backend.main.generate_alert") as mock_alert:
        mock_alert.return_value = {"level": "LOW", "sent": False}
        response = client.get("/test-email-alert?count=10&zone_name=Test&max_capacity=25")
        assert response.status_code == 200
        assert response.json()["level"] == "LOW"

@patch("src.backend.main.process_image")
@patch("src.backend.main.generate_alert")
def test_live_density_endpoint(mock_alert, mock_process):
    mock_process.return_value = {"count": 5, "density_map": "mock_data"}
    mock_alert.return_value = {"level": "LOW", "alert": False}
    
    # Create a mock image file
    file_content = b"fake image content"
    response = client.post(
        "/live-density",
        files={"file": ("test.jpg", file_content, "image/jpeg")},
        data={"zone_name": "Entrance", "max_capacity": "50"}
    )
    
    assert response.status_code == 200
    assert response.json()["count"] == 5
    assert response.json()["risk"]["level"] == "LOW"

def test_predict_zone_endpoint():
    with patch("src.backend.main.get_predictive_risk") as mock_risk:
        mock_risk.return_value = {"status": "ok"}
        response = client.post(
            "/predict-zone",
            json={"periods": 15, "history_counts": [1, 2, 3]}
        )
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
