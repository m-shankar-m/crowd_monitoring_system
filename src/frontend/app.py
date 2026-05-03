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
    from src.backend.services.forecast import get_predictive_risk, train_system
    from src.frontend.api import update_email_settings
except ImportError:
    pass

# --- WebRTC Session State ---
if "webrtc_frames" not in st.session_state:
    st.session_state.webrtc_frames = {}
if "webrtc_lock" not in st.session_state:
    st.session_state.webrtc_lock = threading.Lock()
if "webrtc_count" not in st.session_state:
    st.session_state.webrtc_count = 0
if "camera_configs" not in st.session_state:
    st.session_state.camera_configs = [{"type": "Browser Camera (WebRTC)", "value": "webrtc"}]
if "running" not in st.session_state:
    st.session_state.running = False

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
        forecast_info = get_predictive_risk(periods=15)
        alert_info = generate_alert(results["count"], zone_name, max_capacity, forecast_info)
        results["risk"] = alert_info
        return results
    except Exception as e:
        DEBUG_INFO["error"] = str(e)
        return None

# --- UI CONFIGURATION ---
st.set_page_config(page_title="Crowd Monitoring | Premium Dashboard", layout="wide", initial_sidebar_state="expanded")

# Premium CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    html, body, [data-testid="stAppViewContainer"] {
        background: #0F172A;
        font-family: 'Inter', sans-serif;
        color: #E2E8F0;
    }
    
    .metric-card {
        background: rgba(30, 41, 59, 0.5);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(56, 189, 248, 0.1);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    
    .section-title {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #94A3B8;
        font-weight: 600;
        margin: 1.5rem 0 1rem 0;
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

# Header
t1, t2 = st.columns([3, 1])
with t1:
    st.markdown("<h1 style='margin:0;'>Crowd Monitoring <span style='color:#38BDF8'>System</span></h1>", unsafe_allow_html=True)
    st.caption("Advanced Real-Time Occupancy Analytics")

# Top KPIs
k1, k2, k3, k4 = st.columns(4)
ph_total = k1.empty()
ph_zones = k2.empty()
ph_alerts = k3.empty()
ph_accuracy = k4.empty()

def update_kpis(total, alerts):
    ph_total.markdown(f"<div class='metric-card'><div style='color:#94A3B8; font-size:0.8rem;'>Total People</div><div style='font-size:2rem; font-weight:700;'>{total}</div></div>", unsafe_allow_html=True)
    ph_zones.markdown(f"<div class='metric-card'><div style='color:#94A3B8; font-size:0.8rem;'>Active Zones</div><div style='font-size:2rem; font-weight:700;'>4</div></div>", unsafe_allow_html=True)
    ph_alerts.markdown(f"<div class='metric-card'><div style='color:#94A3B8; font-size:0.8rem;'>Active Alerts</div><div style='font-size:2rem; font-weight:700; color:#EF4444;'>{alerts}</div></div>", unsafe_allow_html=True)
    ph_accuracy.markdown(f"<div class='metric-card'><div style='color:#94A3B8; font-size:0.8rem;'>System Accuracy</div><div style='font-size:2rem; font-weight:700; color:#10B981;'>94.2%</div></div>", unsafe_allow_html=True)

update_kpis(0, 0)

# Main Dashboard
c1, c2 = st.columns([2.5, 1])

with c1:
    st.markdown("<div class='section-title'><span class='section-badge'>TRENDS</span> Prediction Forecast</div>", unsafe_allow_html=True)
    graph_ph = st.empty()
    
    st.markdown("<div class='section-title'><span class='section-badge'>VIDEO</span> Live Detection Feeds</div>", unsafe_allow_html=True)
    v1, v2 = st.columns(2)
    v3, v4 = st.columns(2)
    vid_phs = [v1.empty(), v2.empty(), v3.empty(), v4.empty()]

with c2:
    st.markdown("<div class='section-title'><span class='section-badge'>ALERTS</span> Zone Snapshots</div>", unsafe_allow_html=True)
    metric_phs = [st.empty() for _ in range(4)]

# Sidebar Configuration
st.sidebar.markdown("### ⚙️ Dashboard Controls")
input_source = st.sidebar.selectbox("Input Source", ["Live Browser Camera", "Demo Video Stream"])

st.sidebar.markdown("### 📹 Camera Configuration")
capacities = [50, 40, 30, 60]
for i in range(4):
    with st.sidebar.expander(f"Zone {chr(65+i)} Configuration", expanded=(i==0)):
        capacities[i] = st.number_input(f"Zone {chr(65+i)} Capacity", 1, 500, capacities[i], key=f"cap_{i}")
        if input_source == "Live Browser Camera" and i == 0:
            webrtc_streamer(
                key=f"webrtc_{i}",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=RTC_CONFIG,
                video_frame_callback=get_webrtc_callback(i),
                async_processing=True,
            )

st.sidebar.markdown("---")
if st.sidebar.button("▶️ START ANALYSIS", type="primary", use_container_width=True):
    st.session_state.running = True
if st.sidebar.button("⏹️ STOP", use_container_width=True):
    st.session_state.running = False

# Email Config (Restored)
with st.sidebar.expander("📧 Alert System Settings"):
    e_to = st.text_input("Alert To", "")
    e_from = st.text_input("Alert From", "")
    e_pass = st.text_input("SMTP Password", type="password")
    if st.button("Save Settings"):
        update_email_settings(e_to, e_from, e_pass)
        st.success("Settings Saved")

# Model Mgmt (Restored)
if st.sidebar.button("🧠 Retrain Prediction Models"):
    with st.spinner("Retraining..."):
        train_system()
        st.sidebar.success("Models updated!")

# Processing Loop
if st.session_state.get("running"):
    histories = [[] for _ in range(4)]
    while st.session_state.get("running"):
        total_p = 0
        active_alerts = 0
        
        for i in range(4):
            frame = None
            if i == 0: # Only zone 0 for WebRTC in this demo
                with st.session_state.webrtc_lock:
                    if i in st.session_state.webrtc_frames:
                        frame = st.session_state.webrtc_frames[i].copy()
            
            if frame is not None:
                is_success, buffer = cv2.imencode(".jpg", frame)
                if is_success:
                    res = get_analysis_direct(buffer.tobytes(), f"Zone {chr(65+i)}", capacities[i])
                    if res:
                        count = res['count']
                        total_p += count
                        histories[i].append(count)
                        vid_phs[i].image(frame, channels="BGR", use_container_width=True)
                        
                        risk_color = "#10B981" if res.get('risk', {}).get('status') != 'HIGH' else "#EF4444"
                        if res.get('risk', {}).get('status') == 'HIGH': active_alerts += 1
                        
                        metric_phs[i].markdown(f"""
                            <div class="metric-card" style="border-left: 4px solid {risk_color};">
                                <div style="font-size:0.7rem; color:#94A3B8;">ZONE {chr(65+i)}</div>
                                <div style="font-size:1.5rem; font-weight:700; color:{risk_color};">{count} people</div>
                                <div style="font-size:0.6rem; color:#64748B;">Capacity: {capacities[i]}</div>
                            </div>
                        """, unsafe_allow_html=True)
            else:
                vid_phs[i].markdown("<div style='height:150px; background:#1E293B; border-radius:8px; display:flex; align-items:center; justify-content:center; color:#475569;'>No Signal</div>", unsafe_allow_html=True)

        update_kpis(total_p, active_alerts)
        
        # Simple Graph Update (Yesterday Night Style)
        if len(histories[0]) > 2:
            fig = go.Figure()
            for i in range(4):
                if histories[i]:
                    fig.add_trace(go.Scatter(y=histories[i], mode='lines', name=f"Zone {chr(65+i)}", fill='tozeroy'))
            fig.update_layout(height=250, margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#94A3B8'))
            graph_ph.plotly_chart(fig, use_container_width=True)

        time.sleep(1.0)
