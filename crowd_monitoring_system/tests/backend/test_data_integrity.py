import pytest
from src.risk.alert import generate_alert, predict_future_risk
from src.risk.threshold import get_risk_level

def test_risk_level_logic():
    # Assuming max_capacity = 25
    # Low: < 50% (12.5)
    # Moderate: 50% - 80% (12.5 - 20)
    # High: > 80% (20)
    assert get_risk_level(10, 25) == "LOW"
    assert get_risk_level(15, 25) == "MODERATE"
    assert get_risk_level(22, 25) == "HIGH ALERT"

def test_generate_alert_structure():
    result = generate_alert(22, zone_name="TestZone", max_capacity=25)
    assert result["level"] == "HIGH ALERT"
    assert result["zone"] == "TestZone"
    assert "email" in result
    assert result["alert"] is True

def test_predict_future_risk():
    forecast_data = [
        {"timestamp": "2024-01-01 10:00", "predicted_count": 10},
        {"timestamp": "2024-01-01 10:05", "predicted_count": 22},
        {"timestamp": "2024-01-01 10:10", "predicted_count": 15},
    ]
    risk = predict_future_risk(forecast_data, max_capacity=25)
    assert risk["peak_prediction"] == 22
    assert risk["first_high_risk_time"] == "2024-01-01 10:05"

def test_generate_alert_with_forecast():
    forecast_info = {
        "peak_prediction": 30,
        "first_high_risk_time": "12:00"
    }
    result = generate_alert(15, zone_name="ForecastZone", max_capacity=25, forecast_info=forecast_info)
    assert result["level"] == "MODERATE"
    assert "email" in result

from unittest.mock import patch, MagicMock

def test_alert_email_exception_handling():
    with patch("src.risk.alert._send_high_alert_email") as mock_send:
        # Test generic exception
        mock_send.side_effect = Exception("Test Error")
        result = generate_alert(30, zone_name="ErrorZone", max_capacity=25)
        assert result["email"]["sent"] is False
        assert "email_error" in result["email"]["reason"]
        
        # Test authentication failure specifically
        mock_send.side_effect = Exception("535 Authentication failed")
        result = generate_alert(30, zone_name="AuthErrorZone", max_capacity=25)
        assert "authentication_failed" in result["email"]["reason"]

def test_send_high_alert_email_logic():
    from src.risk.alert import _send_high_alert_email
    
    # Test case: missing credentials
    with patch("os.getenv") as mock_env:
        # Return default value if not found in our mock logic
        def side_effect(key, default=None):
            if key in ["ALERT_EMAIL_FROM", "ALERT_EMAIL_PASSWORD"]:
                return None
            return default
        mock_env.side_effect = side_effect
        result = _send_high_alert_email(30, "msg")
        assert result["sent"] is False
        assert result["reason"] == "email_not_configured"

    # Test case: SMTP logic (mocking smtplib)
    with patch("os.getenv") as mock_env:
        def side_effect_smtp(key, default=None):
            vals = {
                "ALERT_EMAIL_FROM": "from@test.com",
                "ALERT_EMAIL_PASSWORD": "pass",
                "ALERT_SMTP_PORT": "587"
            }
            return vals.get(key, default)
        mock_env.side_effect = side_effect_smtp
        
        with patch("smtplib.SMTP") as mock_smtp:
            # We also need to mock _last_high_alert_ts_by_zone to avoid cooldown
            with patch("src.risk.alert._last_high_alert_ts_by_zone", {}):
                result = _send_high_alert_email(30, "msg", zone_name="SMTPTest")
                assert result["sent"] is True
                assert mock_smtp.called
