import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

session = requests.Session()
retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
session.mount('http://', HTTPAdapter(max_retries=retries))
session.mount('https://', HTTPAdapter(max_retries=retries))

import os

BASE_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000").strip().rstrip("/")

def upload_frame(file_bytes, zone_name=None, max_capacity=25):
    try:
        files = {"file": ("frame.jpg", file_bytes, "image/jpeg")}
        params = {"zone_name": zone_name, "max_capacity": max_capacity}
        # Increased timeout to 10s for slow CPU-based inference on Hugging Face
        resp = session.post(f"{BASE_URL}/live-density", files=files, params=params, timeout=10)
        if resp.status_code == 200:
            return resp.json()
        return {"error": f"Status {resp.status_code}", "detail": resp.text[:100]}
    except Exception as e:
        return {"error": "Connection Failed", "detail": str(e)[:100]}

def train_model():
    try:
        resp = session.post(f"{BASE_URL}/train", timeout=300)
        if resp.status_code == 200:
            return resp.json()
        return {"status": "error", "message": f"Server Error {resp.status_code}: {resp.text[:50]}"}
    except requests.exceptions.Timeout:
        return {"status": "error", "message": "Training timed out (took > 5 mins)"}
    except Exception as e:
        return {"status": "error", "message": f"Connection Failed: {str(e)[:50]}"}

def get_forecast():
    try:
        resp = session.get(f"{BASE_URL}/predict-risk", timeout=5)
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
        resp = session.post(f"{BASE_URL}/update-email-settings", json=data, timeout=5)
        return resp.status_code == 200
    except:
        return False
