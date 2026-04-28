import pytest
import numpy as np
import cv2
from unittest.mock import patch, MagicMock
from src.backend.services.density import process_image
from src.backend.services.forecast import train_system, get_predictions, get_predictive_risk

def test_process_image_valid():
    # Create a real image encoded as bytes
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    _, buffer = cv2.imencode(".jpg", frame)
    image_bytes = buffer.tobytes()
    
    with patch("src.backend.services.density.cv_pipeline") as mock_cv:
        mock_cv.process_frame.return_value = {"count": 10}
        result = process_image(image_bytes)
        assert result["count"] == 10

def test_process_image_invalid():
    with pytest.raises(ValueError, match="Invalid image"):
        process_image(b"not an image")

def test_train_system_service():
    with patch("src.backend.services.forecast.ml_pipeline") as mock_ml:
        mock_ml.train_model.return_value = {"status": "success"}
        assert train_system() == {"status": "success"}

def test_get_predictions_service():
    with patch("src.backend.services.forecast.ml_pipeline") as mock_ml:
        mock_ml.get_forecast.return_value = [{"y": 1}]
        assert get_predictions(10) == [{"y": 1}]

def test_get_predictive_risk_service():
    with patch("src.backend.services.forecast.ml_pipeline") as mock_ml:
        # Test case: no forecasts
        mock_ml.get_forecast.return_value = []
        result = get_predictive_risk()
        assert result["risk"] == "Unknown"
        
        # Test case: with forecasts
        mock_ml.get_forecast.return_value = [{"predicted_count": 10}]
        with patch("src.backend.services.forecast.predict_future_risk") as mock_risk:
            mock_risk.return_value = {"first_high_risk_time": None, "peak_prediction": 10}
            result = get_predictive_risk()
            assert result["peak_prediction"] == 10
