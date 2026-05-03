import streamlit as st
import sys
import os
import time
import random
import threading
import traceback
import cv2
import numpy as np
import plotly.graph_objects as go
import pandas as pd
from PIL import Image
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# --- GLOBAL SETTINGS & PATHS ---
try:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
except:
    pass

# Import backend logic
try:
    from src.backend.services.density import process_image
    from src.risk.alert import generate_alert
    from src.backend.services.forecast import get_predictive_risk
except ImportError:
    st.error("Backend modules not found. Check project structure.")

# --- WebRTC Session State ---
if "webrtc_frames" not in st.session_state:
    st.session_state.webrtc_frames = {}
if "webrtc_lock" not in st.session_state:
    st.session_state.webrtc_lock = threading.Lock()
if "webrtc_count" not in st.session_state:
    st.session_state.webrtc_count = 0

def get_webrtc_callback(zone_idx):
    def callback(frame):
        img = frame.to_ndarray(format="bgr24")
        with st.session_state.webrtc_lock:
            st.session_state.webrtc_frames[zone_idx] = img
            st.session_state.webrtc_count += 1
        return frame
    return callback

RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

DEBUG_INFO = {"error": None, "last_call": None}

def get_analysis_direct(image_bytes, zone_name, max_capacity):
    DEBUG_INFO["last_call"] = f"Zone {zone_name} @ {time.strftime('%H:%M:%S')}"
    try:
        results = process_image(image_bytes, zone_name=zone_name)
        forecast_info = None
        if results.get("count", 0) > 0.5 * max_capacity:
            forecast_info = get_predictive_risk(periods=15)
        alert_info = generate_alert(results["count"], zone_name, max_capacity, forecast_info)
        results["risk"] = alert_info
        return results
    except Exception as e:
        DEBUG_INFO["error"] = str(e)
        return None

# --- UI CONFIGURATION (PREMIUM) ---
st.set_page_config(page_title="Crowd Monitoring | Dashboard", layout="wide", initial_sidebar_state="expanded")

# Premium CSS (Yesterday Night Version)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    html, body, [data-testid="stAppViewContainer"] {
        background: #0F172A;
        font-family: 'Inter', sans-serif;
        color: #E2E8F0;
    }
    
    .stMetric {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(56, 189, 248, 0.1);
        border-radius: 12px;
        padding: 20px;
    }
    
    .metric-card {
        background: rgba(30, 41, 59, 0.5);
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    .section-title {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #94A3B8;
        font-weight: 600;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .section-badge {
        background: rgba(56, 189, 248, 0.15);
        color: #38BDF8;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.7rem;
    }
    </style>
""", unsafe_allow_html=True)

# Top Header
col_title, col_status = st.columns([3, 1])
with col_title:
    st.markdown("<h1 style='margin:0; font-weight:700;'>Crowd Monitoring <span style='color:#38BDF8'>System</span></h1>", unsafe_allow_html=True)
    st.caption("AI-Powered Real-Time Occupancy Analytics & Prediction")

# Layout
c1, c2 = st.columns([2.5, 1])

with c1:
    st.markdown("<div class='section-title'><span class='section-badge'>LIVE</span> Prediction Forecast</div>", unsafe_allow_html=True)
    graph_ph = st.empty()
    
    st.markdown("<div class='section-title'><span class='section-badge'>VIDEO</span> Detection Feeds</div>", unsafe_allow_html=True)
    v1, v2 = st.columns(2)
    v3, v4 = st.columns(2)
    vid_phs = [v1.empty(), v2.empty(), v3.empty(), v4.empty()]

with c2:
    st.markdown("<div class='section-title'><span class='section-badge'>METRICS</span> Zone Snapshots</div>", unsafe_allow_html=True)
    metric_phs = [st.empty() for _ in range(4)]

# Sidebar (Yesterday Night Style)
st.sidebar.markdown("### ⚙️ Dashboard Controls")
input_mode = st.sidebar.selectbox("Analysis Source", ["Live Browser Camera", "Demo Simulation"])

st.sidebar.markdown("### 📹 Camera Configuration")
if input_mode == "Live Browser Camera":
    webrtc_streamer(
        key="webrtc_main",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIG,
        video_frame_callback=get_webrtc_callback(0),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

if st.sidebar.button("▶️ START ANALYSIS", type="primary", use_container_width=True):
    st.session_state.running = True
if st.sidebar.button("⏹️ STOP", use_container_width=True):
    st.session_state.running = False

# Processing Loop
if st.session_state.get("running"):
    while st.session_state.get("running"):
        frame = None
        with st.session_state.webrtc_lock:
            if 0 in st.session_state.webrtc_frames:
                frame = st.session_state.webrtc_frames[0].copy()
        
        if frame is not None:
            is_success, buffer = cv2.imencode(".jpg", frame)
            if is_success:
                res = get_analysis_direct(buffer.tobytes(), "Zone A/1", 50)
                if res:
                    vid_phs[0].image(frame, channels="BGR", use_container_width=True)
                    metric_phs[0].markdown(f"""
                        <div class="metric-card">
                            <div style="font-size:0.8rem; color:#94A3B8;">Zone A/1</div>
                            <div style="font-size:1.8rem; font-weight:700; color:#38BDF8;">{res['count']} <span style='font-size:1rem; color:#64748B;'>people</span></div>
                        </div>
                    """, unsafe_allow_html=True)
        
        time.sleep(0.5)
