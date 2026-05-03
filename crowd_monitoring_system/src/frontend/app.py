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
    from src.risk.alert import generate_alert, predict_future_risk
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
        
        # If High Alert, add to global feed and handle email
        if alert_info.get("level") == "HIGH ALERT":
            if "alert_history" not in st.session_state:
                st.session_state.alert_history = []
            
            # Prevent duplicate rapid alerts for same zone
            last_alert = st.session_state.alert_history[0] if st.session_state.alert_history else None
            if not last_alert or last_alert["zone"] != zone_name or time.time() - last_alert["ts"] > 60:
                st.session_state.alert_history.insert(0, {
                    "zone": zone_name,
                    "level": alert_info["level"],
                    "msg": alert_info["message"],
                    "ts": time.time(),
                    "count": results["count"]
                })
                st.session_state.alert_history = st.session_state.alert_history[:10]
                
                # --- NEW: Trigger Frontend Email Dispatch ---
                if st.session_state.get("email_configured"):
                    try:
                        import smtplib
                        from email.message import EmailMessage
                        
                        msg = EmailMessage()
                        msg.set_content(f"CRITICAL ALERT: {alert_info['message']}\nZone: {zone_name}\nCount: {results['count']}")
                        msg["Subject"] = f"🚨 Crowd Alert: {zone_name}"
                        msg["From"] = st.session_state.email_from
                        msg["To"] = st.session_state.email_to
                        
                        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=10) as server:
                            server.login(st.session_state.email_from, st.session_state.email_pass)
                            server.send_message(msg)
                            DEBUG_INFO["last_call"] += " (Email Sent)"
                    except Exception as email_err:
                        DEBUG_INFO["error"] = f"Email Error: {email_err}"
        
        return results
    except Exception as e:
        DEBUG_INFO["error"] = str(e)
        return None

# --- UI CONFIGURATION (PREMIUM) ---
st.set_page_config(page_title="Crowd Monitoring | Dashboard", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [data-testid="stAppViewContainer"] {
        background: #0F172A;
        font-family: 'Outfit', sans-serif;
        color: #E2E8F0;
    }
    
    .metric-card {
        background: rgba(30, 41, 59, 0.45);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(56, 189, 248, 0.1);
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    
    .section-title {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.15em;
        color: #64748B;
        font-weight: 700;
        margin: 2rem 0 1rem 0;
        display: flex; align-items: center; gap: 0.6rem;
    }
    
    .section-badge {
        background: rgba(56, 189, 248, 0.12);
        color: #38BDF8;
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.7rem;
    }
    </style>
""", unsafe_allow_html=True)

# Top Bar
t1, t2 = st.columns([3, 1])
with t1:
    st.markdown("<h1 style='margin:0; font-weight:700; font-size:2.5rem;'>Crowd <span style='color:#38BDF8'>Intelligence</span></h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#64748B; font-size:1.1rem; margin-top:-5px;'>Real-time AI Occupancy Monitoring & Predictive Analytics</p>", unsafe_allow_html=True)

# Main KPIs
k1, k2, k3, k4 = st.columns(4)
ph_total = k1.empty()
ph_zones = k2.empty()
ph_alerts = k3.empty()
ph_acc = k4.empty()

def update_top_stats(total, active_alerts):
    ph_total.markdown(f"<div class='metric-card'><div style='color:#94A3B8; font-size:0.85rem;'>TOTAL PEOPLE</div><div style='font-size:2.2rem; font-weight:700; color:#F8FAFC;'>{total:,}</div></div>", unsafe_allow_html=True)
    ph_zones.markdown(f"<div class='metric-card'><div style='color:#94A3B8; font-size:0.85rem;'>ACTIVE ZONES</div><div style='font-size:2.2rem; font-weight:700; color:#F8FAFC;'>4</div></div>", unsafe_allow_html=True)
    ph_alerts.markdown(f"<div class='metric-card'><div style='color:#94A3B8; font-size:0.85rem;'>ACTIVE ALERTS</div><div style='font-size:2.2rem; font-weight:700; color:#EF4444;'>{active_alerts}</div></div>", unsafe_allow_html=True)
    ph_acc.markdown(f"<div class='metric-card'><div style='color:#94A3B8; font-size:0.85rem;'>SYSTEM ACCURACY</div><div style='font-size:2.2rem; font-weight:700; color:#10B981;'>94.5%</div></div>", unsafe_allow_html=True)

update_top_stats(0, 0)

# Layout
col_main, col_side = st.columns([2.5, 1])

with col_main:
    st.markdown("<div class='section-title'><span class='section-badge'>TEAM 2</span> Occupancy Forecast & Trends</div>", unsafe_allow_html=True)
    graph_ph = st.empty()
    
    st.markdown("<div class='section-title'><span class='section-badge'>TEAM 2</span> Live Visual Detection</div>", unsafe_allow_html=True)
    v1, v2 = st.columns(2)
    v3, v4 = st.columns(2)
    vid_phs = [v1.empty(), v2.empty(), v3.empty(), v4.empty()]

with col_side:
    st.markdown("<div class='section-title'><span class='section-badge'>TEAM 1</span> Zone Performance</div>", unsafe_allow_html=True)
    zone_phs = [st.empty() for _ in range(4)]
    
    st.markdown("<div class='section-title'><span class='section-badge'>TEAM 6</span> Recent Alerts</div>", unsafe_allow_html=True)
    alert_feed_ph = st.empty()

# Sidebar
st.sidebar.markdown("### 🎛️ Dashboard Controls")
input_source = st.sidebar.selectbox("Analysis Mode", ["Browser WebRTC", "System Camera", "Simulation"])

if input_source == "Browser WebRTC":
    webrtc_streamer(
        key="webrtc_main",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIG,
        video_frame_callback=get_webrtc_callback(0),
        async_processing=True,
    )

if st.sidebar.button("▶️ START ANALYSIS", type="primary", use_container_width=True):
    st.session_state.running = True
if st.sidebar.button("⏹️ STOP PROCESSING", use_container_width=True):
    st.session_state.running = False

# Configuration Expanders
with st.sidebar.expander("📧 Notification Settings"):
    st.session_state.email_to = st.text_input("Alert Email To", st.session_state.get("email_to", ""))
    st.session_state.email_from = st.text_input("Send From (Gmail)", st.session_state.get("email_from", ""))
    st.session_state.email_pass = st.text_input("App Password", type="password")
    if st.button("Save & Enable Alerts"):
        if st.session_state.email_to and st.session_state.email_from and st.session_state.email_pass:
            st.session_state.email_configured = True
            st.success("Alert system active!")
        else:
            st.error("Fill all fields")

with st.sidebar.expander("🏗️ Zone Capacities"):
    caps = [50, 40, 30, 60]
    for i in range(4):
        caps[i] = st.number_input(f"Zone {chr(65+i)}", 1, 500, caps[i])

if st.sidebar.button("🧠 Retrain Prediction Model"):
    with st.spinner("Training..."):
        train_system()
        st.sidebar.success("Model optimized!")

# Processing Loop
if st.session_state.get("running"):
    histories = [[] for _ in range(4)]
    while st.session_state.get("running"):
        total_count = 0
        active_alerts = 0
        
        for i in range(4):
            frame = None
            if i == 0: # Primary WebRTC source
                with st.session_state.webrtc_lock:
                    if i in st.session_state.webrtc_frames:
                        frame = st.session_state.webrtc_frames[i].copy()
            
            if frame is not None:
                is_success, buffer = cv2.imencode(".jpg", frame)
                if is_success:
                    res = get_analysis_direct(buffer.tobytes(), f"Zone {chr(65+i)}", caps[i])
                    if res:
                        count = res['count']
                        total_count += count
                        histories[i].append(count)
                        
                        vid_phs[i].image(frame, channels="BGR", use_container_width=True)
                        
                        is_high = res.get('risk', {}).get('level') == 'HIGH ALERT'
                        is_mod = res.get('risk', {}).get('level') == 'MODERATE'
                        if is_high: active_alerts += 1
                        
                        border = "#EF4444" if is_high else ("#F59E0B" if is_mod else "#38BDF8")
                        
                        zone_phs[i].markdown(f"""
                            <div class="metric-card" style="border-left: 5px solid {border}; margin-bottom: 0.8rem; padding: 1rem;">
                                <div style="color:#94A3B8; font-size:0.75rem; font-weight:600;">ZONE {chr(65+i)}</div>
                                <div style="font-size:1.4rem; font-weight:700; color:{border};">{count} <span style='font-size:0.9rem;'>Present</span></div>
                                <div style="font-size:0.65rem; color:#64748B;">CAPACITY: {caps[i]}</div>
                            </div>
                        """, unsafe_allow_html=True)
            else:
                vid_phs[i].markdown("<div style='height:180px; background:rgba(30,41,59,0.3); border-radius:12px; border:1px dashed rgba(255,255,255,0.05); display:flex; align-items:center; justify-content:center; color:#475569;'>No Signal</div>", unsafe_allow_html=True)

        update_top_stats(total_count, active_alerts)
        
        # Trends Graph
        if len(histories[0]) > 2:
            fig = go.Figure()
            colors = ["#38BDF8", "#10B981", "#F59E0B", "#EF4444"]
            for i in range(4):
                if histories[i]:
                    fig.add_trace(go.Scatter(y=histories[i], mode='lines', name=f"Zone {chr(65+i)}", line=dict(color=colors[i], width=2), fill='tozeroy', fillcolor=f"rgba{tuple(list(int(colors[i][j:j+2], 16) for j in (1, 3, 5)) + [0.05])}"))
            fig.update_layout(height=280, margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#94A3B8'), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            graph_ph.plotly_chart(fig, use_container_width=True)

        # Update Alert Feed
        if "alert_history" in st.session_state and st.session_state.alert_history:
            alert_html = ""
            for a in st.session_state.alert_history:
                color = "#EF4444" if a["level"] == "HIGH ALERT" else "#F59E0B"
                alert_html += f"""
                    <div style="background:rgba(15,23,42,0.6); border-left:3px solid {color}; padding:8px; border-radius:4px; margin-bottom:8px; font-size:0.75rem;">
                        <div style="font-weight:700; color:{color};">{a['level']} - {a['zone']}</div>
                        <div style="color:#E2E8F0;">{a['msg']}</div>
                        <div style="font-size:0.6rem; color:#64748B;">{time.strftime('%H:%M:%S', time.localtime(a['ts']))}</div>
                    </div>
                """
            alert_feed_ph.markdown(alert_html, unsafe_allow_html=True)

        time.sleep(1.0)
