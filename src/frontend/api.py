import requests

import os

BASE_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000").strip().rstrip("/")

def upload_frame(file_bytes, zone_name=None, max_capacity=25):
    try:
        files = {"file": ("frame.jpg", file_bytes, "image/jpeg")}
        params = {"zone_name": zone_name, "max_capacity": max_capacity}
        # Increased timeout to 10s for slow CPU-based inference on Hugging Face
        resp = requests.post(f"{BASE_URL}/live-density", files=files, params=params, timeout=10)
        if resp.status_code == 200:
            return resp.json()
        return {"error": f"Status {resp.status_code}", "detail": resp.text[:100]}
    except Exception as e:
        return {"error": "Connection Failed", "detail": str(e)[:100]}

def train_model():
    try:
        resp = requests.post(f"{BASE_URL}/train", timeout=300)
        if resp.status_code == 200:
            return resp.json()
        return {"status": "error", "message": f"Server Error {resp.status_code}: {resp.text[:50]}"}
    except requests.exceptions.Timeout:
        return {"status": "error", "message": "Training timed out (took > 5 mins)"}
    except Exception as e:
        return {"status": "error", "message": f"Connection Failed: {str(e)[:50]}"}

def get_forecast():
    try:
        resp = requests.get(f"{BASE_URL}/predict-risk", timeout=5)
        return resp.json()
    except:
        return None

def update_email_settings(email_to, email_from, email_password):
    try:
        data = {
            "email_to": email_to,
            "email_from": email_from,
            "email_password": email_password
        }
        resp = requests.post(f"{BASE_URL}/update-email-settings", json=data, timeout=5)
        return resp.status_code == 200
    except:
        return False
