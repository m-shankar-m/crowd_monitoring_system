from .threshold import get_risk_level
import os
import time
import smtplib
import ssl
from email.message import EmailMessage
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

LOG_PATH = "logs/alerts.log"
_last_high_alert_ts_by_zone = {}

def _send_high_alert_email(count, message, zone_name="Unknown", forecast_info=None):
    global _last_high_alert_ts_by_zone

    email_to = os.getenv("ALERT_EMAIL_TO", "shankarm1612@gmail.com")
    email_from = os.getenv("ALERT_EMAIL_FROM")
    email_password = os.getenv("ALERT_EMAIL_PASSWORD")
    smtp_host = os.getenv("ALERT_SMTP_HOST", "smtp.gmail.com")
    smtp_port = int(os.getenv("ALERT_SMTP_PORT", "465"))
    email_cooldown_seconds = int(os.getenv("ALERT_EMAIL_COOLDOWN_SECONDS", "60"))

    # Skip email when SMTP credentials are not configured.
    if not email_from or not email_password:
        return {
            "sent": False,
            "reason": "email_not_configured"
        }

    now = int(time.time())
    last_ts = _last_high_alert_ts_by_zone.get(zone_name, 0)
    
    if now - last_ts < email_cooldown_seconds:
        return {
            "sent": False,
            "reason": "cooldown_active"
        }

    _last_high_alert_ts_by_zone[zone_name] = now
    
    subject = f"High Crowd Alert - {zone_name} (Count {count})"
    body = (
        f"Critical crowd density threshold has been exceeded in {zone_name}.\n\n"
        f"Zone: {zone_name}\n"
        f"Risk Level: HIGH ALERT\n"
        f"Count: {count}\n"
        f"Message: {message}\n"
        f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
    )
    
    if forecast_info:
        body += (
            f"\n--- FORECAST OUTLOOK ---\n"
            f"Peak Expected: {forecast_info.get('peak_prediction', 'N/A')} people\n"
            f"High Risk Persists Until: {forecast_info.get('first_high_risk_time', 'N/A')}\n"
            f"Recommendation: Immediate staff deployment suggested.\n"
        )

    email = EmailMessage()
    email["From"] = email_from
    email["To"] = email_to
    email["Subject"] = subject
    email.set_content(body)

    if smtp_port == 465:
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(smtp_host, smtp_port, context=context, timeout=15) as smtp:
            smtp.login(email_from, email_password)
            smtp.send_message(email)
    else:
        with smtplib.SMTP(smtp_host, smtp_port, timeout=15) as smtp:
            smtp.ehlo()
            smtp.starttls(context=ssl.create_default_context())
            smtp.ehlo()
            smtp.login(email_from, email_password)
            smtp.send_message(email)

    return {
        "sent": True,
        "reason": "sent"
    }

def generate_alert(count, zone_name="Unknown", max_capacity=25, forecast_info=None):
    level = get_risk_level(count, max_capacity)
    is_alert = level in ["MODERATE", "HIGH ALERT"]
    
    if level == "LOW":
        message = f"Crowd levels in {zone_name} are normal."
    elif level == "MODERATE":
        message = f"Warning: Crowd density in {zone_name} is rising."
    else:
        message = f"CRITICAL: High crowd density detected in {zone_name}!"
        
    if is_alert:
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        with open(LOG_PATH, "a") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {level} [{zone_name}]: {message} (Count: {count})\n")

    email_status = {"sent": False, "reason": "not_high_alert"}
    if level == "HIGH ALERT":
        try:
            email_status = _send_high_alert_email(count, message, zone_name, forecast_info)
        except Exception as exc:
            reason = str(exc)
            if "BadCredentials" in reason or "535" in reason:
                reason = "authentication_failed (Check Gmail App Password and 2FA status)"
            elif "Connection unexpectedly closed" in reason:
                reason = "connection_error (Possible rate limiting or local network block)"
            email_status = {"sent": False, "reason": f"email_error: {reason}"}
            
        with open(LOG_PATH, "a") as f:
            f.write(
                f"{time.strftime('%Y-%m-%d %H:%M:%S')} - EMAIL_STATUS [{zone_name}]: "
                f"{email_status.get('reason')} (Count: {count})\n"
            )
        
    return {
        "level": level,
        "message": message,
        "alert": is_alert,
        "email": email_status,
        "zone": zone_name
    }

def predict_future_risk(forecast_data, max_capacity=25):
    peak_pred = 0
    first_high_risk = None
    
    for row in forecast_data:
        pred = row.get("predicted_count", 0)
        ts = row.get("timestamp")
        
        if pred > peak_pred:
            peak_pred = pred
            
        if get_risk_level(pred, max_capacity) == "HIGH ALERT" and first_high_risk is None:
            first_high_risk = ts
            
    return {
        "first_high_risk_time": first_high_risk,
        "peak_prediction": peak_pred
    }
