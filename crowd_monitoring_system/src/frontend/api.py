import os
import requests
import streamlit as st

# Check if Streamlit secrets are available, otherwise use environment variable or fallback to local
try:
    if "API_URL" in st.secrets:
        BASE_URL = st.secrets["API_URL"]
    else:
        BASE_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
except Exception:
    BASE_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

def upload_frame(file_bytes, zone_name=None, max_capacity=25):
    try:
        files = {"file": ("frame.jpg", file_bytes, "image/jpeg")}
        params = {"zone_name": zone_name, "max_capacity": max_capacity}
        resp = requests.post(f"{BASE_URL}/live-density", files=files, params=params, timeout=60)
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
    """Update email settings on the backend for alert configuration"""
    try:
        data = {
            "email_to": email_to,
            "email_from": email_from,
            "email_password": email_password
        }
        resp = requests.post(f"{BASE_URL}/update-email-settings", json=data)
        return resp.status_code == 200
    except:
        return False
