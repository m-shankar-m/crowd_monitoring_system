import pytest
from unittest.mock import patch, mock_open
import os
from src.risk.threshold import get_risk_level
from src.risk.alert import generate_alert, predict_future_risk

def test_get_risk_level():
    # max_capacity = 25 (default)
    # LOW: < 15 (60%)
    # MODERATE: 15 - 20 (80%)
    # HIGH ALERT: > 20
    assert get_risk_level(10) == "LOW"
    assert get_risk_level(14.9) == "LOW"
    assert get_risk_level(15) == "MODERATE"
    assert get_risk_level(20) == "MODERATE"
    assert get_risk_level(20.1) == "HIGH ALERT"
    assert get_risk_level(30) == "HIGH ALERT"

def test_predict_future_risk():
    forecast_data = [
        {"timestamp": "2023-01-01 00:00:00", "predicted_count": 10},
        {"timestamp": "2023-01-01 01:00:00", "predicted_count": 22},
        {"timestamp": "2023-01-01 02:00:00", "predicted_count": 18},
    ]
    result = predict_future_risk(forecast_data, max_capacity=25)
    assert result["peak_prediction"] == 22
    assert result["first_high_risk_time"] == "2023-01-01 01:00:00"

    forecast_data_low = [
        {"timestamp": "2023-01-01 00:00:00", "predicted_count": 5},
    ]
    result_low = predict_future_risk(forecast_data_low, max_capacity=25)
    assert result_low["peak_prediction"] == 5
    assert result_low["first_high_risk_time"] is None

def test_generate_alert_low():
    with patch("builtins.open", mock_open()) as mocked_file:
        result = generate_alert(5, zone_name="Zone A", max_capacity=25)
        assert result["level"] == "LOW"
        assert result["alert"] is False
        assert "normal" in result["message"]
        # Should not write to log for LOW (based on src/risk/alert.py logic)
        mocked_file.assert_not_called()

def test_generate_alert_moderate():
    with patch("builtins.open", mock_open()) as mocked_file:
        # We need to mock os.makedirs as well because generate_alert calls it
        with patch("os.makedirs"):
            result = generate_alert(18, zone_name="Zone B", max_capacity=25)
            assert result["level"] == "MODERATE"
            assert result["alert"] is True
            assert "rising" in result["message"]
            # Should write to log
            mocked_file.assert_called()

@patch("src.risk.alert._send_high_alert_email")
def test_generate_alert_high(mock_email):
    mock_email.return_value = {"sent": True, "reason": "sent"}
    with patch("builtins.open", mock_open()) as mocked_file:
        with patch("os.makedirs"):
            result = generate_alert(22, zone_name="Zone C", max_capacity=25)
            assert result["level"] == "HIGH ALERT"
            assert result["alert"] is True
            assert "CRITICAL" in result["message"]
            assert result["email"]["sent"] is True
            # Should write to log
            mocked_file.assert_called()
            mock_email.assert_called_once()
