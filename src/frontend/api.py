import requests

import os

BASE_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

def upload_frame(file_bytes, zone_name=None, max_capacity=25):
    try:
        files = {"file": ("frame.jpg", file_bytes, "image/jpeg")}
        params = {"zone_name": zone_name, "max_capacity": max_capacity}
        resp = requests.post(f"{BASE_URL}/live-density", files=files, params=params, timeout=2)
        return resp.json() if resp.status_code == 200 else None
    except:
        return None

def train_model():
    try:
        resp = requests.post(f"{BASE_URL}/train", timeout=300)
        return resp.json()
    except:
        return None

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
