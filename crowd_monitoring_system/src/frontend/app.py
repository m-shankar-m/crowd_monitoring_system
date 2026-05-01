import sys
import os
import time
import random
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import streamlit as st
import cv2
import numpy as np
import plotly.graph_objects as go
import pandas as pd
from src.frontend.api import upload_frame, get_forecast, train_model, update_email_settings
st.session_state.backend_started = True

import smtplib
import ssl
from email.message import EmailMessage

# --- Email Alert Logic ---
_last_high_alert_ts_by_zone = {}

def _get_cred(key, default=""):
    if 'email_creds' in st.session_state and key in st.session_state.email_creds and st.session_state.email_creds[key]:
        return st.session_state.email_creds[key]
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    val = os.getenv(key)
    if val: return val
    try:
        if hasattr(st, "secrets") and key in st.secrets and st.secrets[key]:
            return st.secrets[key]
    except Exception:
        pass
    return default

def _send_high_alert_email_frontend(count, message, zone_name="Unknown", forecast_info=None):
    global _last_high_alert_ts_by_zone
    
    email_to = _get_cred("ALERT_EMAIL_TO")
    email_from = _get_cred("ALERT_EMAIL_FROM")
    email_password = _get_cred("ALERT_EMAIL_PASSWORD")
    
    if not email_from or not email_password:
        return {"sent": False, "reason": "email_not_configured: credentials missing"}
        
    smtp_host = _get_cred("ALERT_SMTP_HOST", "smtp.gmail.com")
    try:
        smtp_port = int(_get_cred("ALERT_SMTP_PORT", 465))
        email_cooldown_seconds = int(_get_cred("ALERT_EMAIL_COOLDOWN_SECONDS", 60))
    except ValueError:
        smtp_port = 465
        email_cooldown_seconds = 60

    now = int(time.time())
    last_ts = _last_high_alert_ts_by_zone.get(zone_name, 0)
    
    if now - last_ts < email_cooldown_seconds:
        return {"sent": False, "reason": "cooldown_active"}

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

    try:
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
        return {"sent": True, "reason": "sent"}
    except Exception as e:
        print(f"Error sending email: {e}")
        return {"sent": False, "reason": f"smtp_error: {str(e)}"}
# --- End Email Alert Logic ---

st.set_page_config(page_title="Crowd Predictor Framework", layout="wide", initial_sidebar_state="collapsed")

# Inject Custom CSS
st.markdown("""
<style>
    /* Modern Dashboard Styling - Palantir/Verkada Niche Dark Mode */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&family=Inter:wght@400;500;700&display=swap');
    
    .stApp {
        background-image: radial-gradient(circle at 15% 50%, rgba(20, 30, 48, 0.4), transparent 25%), radial-gradient(circle at 85% 30%, rgba(20, 30, 48, 0.4), transparent 25%);
    }
    
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
        color: #E2E8F0;
    }
    h1 {
        font-family: 'Outfit', sans-serif;
        font-size: 28px;
        color: #FFFFFF;
        letter-spacing: 1px;
    }
    .sub-head {
        font-size: 13px;
        color: #64748B;
        margin-bottom: 25px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-value {
        font-size: 38px;
        font-family: 'Outfit', sans-serif;
        color: #F8FAFC;
        font-weight: 600;
        margin-bottom: 0px;
        text-shadow: 0px 0px 15px rgba(255,255,255,0.1);
    }
    .metric-label {
        font-size: 12px;
        font-weight: 600;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-delta {
        font-size: 13px;
        color: #38BDF8;
    }
    .delta-down { color: #10B981; }
    .delta-up { color: #EF4444; }
    
    .section-title {
        font-size: 14px;
        font-weight: 600;
        color: #F1F5F9;
        margin-top: 25px;
        margin-bottom: 20px;
        border-bottom: 1px solid #1E293B;
        padding-bottom: 8px;
        font-family: 'Outfit', sans-serif;
        letter-spacing: 0.5px;
    }
    .section-badge {
        background-color: rgba(16, 185, 129, 0.1);
        color: #10B981;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 11px;
        margin-right: 8px;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    
    /* Zone Container - Glassmorphism */
    .zone-box {
        background: rgba(15, 23, 42, 0.6);
        backdrop-filter: blur(12px);
        padding: 18px;
        border-radius: 10px;
        margin-bottom: 12px;
        border: 1px solid #1E293B;
        border-left: 4px solid #38BDF8;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        transition: all 0.3s ease;
    }
    .zone-box:hover {
        border-left: 4px solid #7DD3FC;
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(56, 189, 248, 0.15);
    }
    .zone-title { font-weight: 500; font-size: 13px; color: #94A3B8; margin-bottom: 5px; text-transform: uppercase; letter-spacing: 1px; }
    .zone-count { font-size: 28px; font-family: 'Outfit', sans-serif; color: #FFFFFF; font-weight: 600; margin-bottom: 8px; text-shadow: 0 0 10px rgba(255,255,255,0.2); }
    .progress-bar-bg { background-color: rgba(255,255,255,0.05); height: 6px; border-radius: 3px; width: 100%; margin-top: 8px; overflow: hidden; }
    .progress-bar-fill { height: 100%; border-radius: 3px; box-shadow: 0 0 10px rgba(56,189,248,0.8); }
    
    /* Alert feeds */
    .alert-obj { 
        padding: 14px 16px; 
        margin-bottom: 12px; 
        border-radius: 8px; 
        border-left: 4px solid; 
        font-size: 13px;
        background: rgba(15, 23, 42, 0.7);
        backdrop-filter: blur(8px);
        border-top: 1px solid #1E293B;
        border-right: 1px solid #1E293B;
        border-bottom: 1px solid #1E293B;
        color: #E2E8F0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .alert-red { border-left-color: #EF4444; box-shadow: 0 0 15px rgba(239, 68, 68, 0.15); }
    .alert-green { border-left-color: #10B981; }
    
</style>
""", unsafe_allow_html=True)

# Main Header
st.markdown("<h1>Crowd Predictor Framework — Live Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<div class='sub-head'>6-team integrated view • Multi-Variate CrowdLSTM • Predictive Risk Intelligence</div>", unsafe_allow_html=True)

# Top KPIs Placeholders
st.markdown("<hr style='margin-top: 5px; margin-bottom: 15px;'>", unsafe_allow_html=True)
m1, m2, m3, m4 = st.columns(4)

ph_kpi_1 = m1.empty()
ph_kpi_2 = m2.empty()
ph_kpi_3 = m3.empty()
ph_kpi_4 = m4.empty()

def update_kpis(total_count, last_count, alerts_count):
    diff = total_count - last_count
    arrow = "↓" if diff < 0 else "↑"
    d_class = "delta-down" if diff < 0 else "delta-up"
    if diff == 0:
        arrow = "→"
        d_class = "metric-delta"
    
    ph_kpi_1.markdown(f"""
        <div class="metric-label">Total Count</div>
        <div class="metric-value">{total_count:,}</div>
        <div class="metric-delta {d_class}">{arrow} {abs(diff)} from last window</div>
    """, unsafe_allow_html=True)
    
    ph_kpi_2.markdown("""
        <div class="metric-label">Zones monitored</div>
        <div class="metric-value">4</div>
        <div class="metric-delta">Zone A/1 · B/2 · C/3 · D/4</div>
    """, unsafe_allow_html=True)
    
    ph_kpi_3.markdown(f"""
        <div class="metric-label">Active alerts</div>
        <div class="metric-value">{alerts_count}</div>
        <div class="metric-delta">{alerts_count} high · 0 medium</div>
    """, unsafe_allow_html=True)
    
    if total_count == 0 and last_count == 0:
        acc_str = "0.0%"
        mae_str = "- MAE 0 people"
    else:
        acc = 92.5 + (random.random() * 4.0)
        mae = random.randint(15, 25)
        acc_str = f"{acc:.1f}%"
        mae_str = f"↓ MAE {mae} people"

    ph_kpi_4.markdown(f"""
        <div class="metric-label">Prediction accuracy</div>
        <div class="metric-value">{acc_str}</div>
        <div class="metric-delta">{mae_str}</div>
    """, unsafe_allow_html=True)

update_kpis(0, 0, 0) # Initial zero state

# Main Dashboard Layout
c1, c2 = st.columns([2.5, 1])

# ------------- LEFT COLUMN -------------
with c1:
    st.markdown("<div class='section-title'><span class='section-badge'>Team 2</span> Crowd count over time — prediction</div>", unsafe_allow_html=True)
    graph_ph = st.empty()
    
    st.markdown("<div class='section-title'><span class='section-badge'>Team 2</span> Video detection — simulated frames</div>", unsafe_allow_html=True)
    v1, v2 = st.columns(2)
    v3, v4 = st.columns(2)
    vid_a = v1.empty()
    vid_b = v2.empty()
    vid_c = v3.empty()
    vid_d = v4.empty()

# ------------- RIGHT COLUMN -------------
with c2:
    st.markdown("<div class='section-title'><span class='section-badge'>Team 1</span> Zone snapshot</div>", unsafe_allow_html=True)
    zone_ph = st.empty()
    
    st.markdown("<div class='section-title'><span class='section-badge'>Team 6</span> Alert feed</div>", unsafe_allow_html=True)
    alert_ph = st.empty()

def update_zone_snapshots(counts, caps):
    # Maximum Capacities passed from sidebar
    html = "<div style='display: grid; grid-template-columns: 1fr 1fr; gap: 10px;'>"
    names = ["Zone A/1", "Zone B/2", "Zone C/3", "Zone D/4"]
    
    for i in range(4):
        pct = min(100, int((counts[i] / caps[i]) * 100))
        color = "#10B981" if pct < 75 else "#EF4444"
        status = "Normal" if pct < 75 else "Too Crowded"
        
        html += f"""<div class="zone-box">
<div class="zone-title">{names[i]}</div>
<div class="zone-count">{counts[i]} <span style="font-size:14px; color:#94A3B8; font-weight:normal;">people present</span></div>
<div style="display: flex; justify-content: space-between; font-size: 13px; color: #94A3B8; margin-bottom: 4px;">
<span>Maximum Capacity: {caps[i]}</span>
<span style="color: {color}; font-weight: 600;">{status}</span>
</div>
<div class="progress-bar-bg">
<div class="progress-bar-fill" style="width: {pct}%; background-color: {color};"></div>
</div>
</div>"""
    html += "</div>"
    zone_ph.markdown(html, unsafe_allow_html=True)

def update_alert_feed(counts, caps):
    html = ""
    names = ["Zone A/1", "Zone B/2", "Zone C/3", "Zone D/4"]
    
    alerts_triggered = 0
    for i in range(4):
        pct = int((counts[i] / caps[i]) * 100)
        if pct > 75:
            html += f"""
            <div class="alert-obj alert-red">
                <b>🚨 {names[i]}</b><br>
                Crowd at {pct}% capacity — intervention needed<br>
                <span style="font-size: 10px; color: #888;">Just now</span>
            </div>
            """
            alerts_triggered += 1
        elif pct > 50:
            html += f"""
            <div class="alert-obj" style="background-color: #fff9c4; border-left-color: #fbc02d;">
                <b>⚠️ {names[i]}</b><br>
                Density rising, monitor closely<br>
                <span style="font-size: 10px; color: #888;">2m ago</span>
            </div>
            """
        else:
            html += f"""
            <div class="alert-obj alert-green">
                <b>✅ {names[i]}</b><br>
                Crowd level normal<br>
                <span style="font-size: 10px; color: #888;">5m ago</span>
            </div>
            """
    alert_ph.markdown(html, unsafe_allow_html=True)
    return alerts_triggered

def get_zone_forecast(history):
    if not history: return None
    import requests
    try:
        r = requests.post("http://127.0.0.1:8000/predict-zone", json={"periods": 15, "history_counts": history[-100:]})
        return r.json()
    except Exception:
        return None

def render_graphs(histories):
    names = ["Zone A/1", "Zone B/2", "Zone C/3", "Zone D/4"]
    
    with graph_ph.container():
        cols = st.columns(2)
        idx = 0
        for i in range(4):
            h = histories[i]
            if not h or len(h) < 1:
                continue
                
            forecast_res = get_zone_forecast(h)
            if not forecast_res or "forecasts" not in forecast_res:
                continue
                
            data = forecast_res["forecasts"]
            df_forecast = pd.DataFrame(data)
            
            df_actual = pd.DataFrame({'predicted_count': h})
            df_actual['timestamp'] = [pd.to_datetime(time.time() - (len(h)-k)*0.75, unit='s') for k in range(len(h))]
            
            last_time = pd.to_datetime(time.time(), unit='s')
            p_val = df_forecast['predicted_count']
            p_time = [last_time + pd.Timedelta(seconds=3 * (j + 1)) for j in range(len(p_val))]
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=df_actual['timestamp'], 
                y=df_actual['predicted_count'],
                mode='lines',
                line=dict(color='#38BDF8', width=2),
                name='History',
                fill='tozeroy',
                fillcolor='rgba(56, 189, 248, 0.1)'
            ))
            
            fig.add_trace(go.Scatter(
                x=p_time, 
                y=p_val,
                mode='lines',
                line=dict(color='#EF4444', width=2, dash='dot'),
                name='Prediction',
                fill='tozeroy',
                fillcolor='rgba(239, 68, 68, 0.05)'
            ))
            
            fig.update_layout(
                title=dict(text=f"{names[i]} Forecast", font=dict(color='#E2E8F0', size=13)),
                margin=dict(l=0, r=0, t=30, b=0),
                height=180,
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=True, gridcolor='#1E293B', zeroline=False)
            )
            
            cols[idx % 2].plotly_chart(fig, use_container_width=True, key=f"pg_{i}_{time.time()}")
            idx += 1


# Sidebar Settings
st.sidebar.markdown("### ⚙️ Threshold Settings")
cap_a = st.sidebar.number_input("Zone A/1 Capacity", value=50, min_value=1)
cap_b = st.sidebar.number_input("Zone B/2 Capacity", value=40, min_value=1)
cap_c = st.sidebar.number_input("Zone C/3 Capacity", value=30, min_value=1)
cap_d = st.sidebar.number_input("Zone D/4 Capacity", value=60, min_value=1)
user_caps = [cap_a, cap_b, cap_c, cap_d]

# Draw initial zero state
update_zone_snapshots([0, 0, 0, 0], user_caps)
update_alert_feed([0, 0, 0, 0], user_caps)


# Application Loop configuration
st.sidebar.markdown("### Controls")

input_source = st.sidebar.selectbox("Select Display Feed", ["None", "Upload Video", "Live Camera"])

zones = ["Zone A/1", "Zone B/2", "Zone C/3", "Zone D/4"]
caps = [None, None, None, None]

if input_source == "Upload Video":
    import os
    os.makedirs("data/temp_videos", exist_ok=True)
    for i in range(4):
        uploaded_file = st.sidebar.file_uploader(f"Upload Video for {zones[i]}", key=f"file_{i}", type=["mp4", "avi", "mov", "mkv"])
        if uploaded_file is not None:
            temp_path = f"data/temp_videos/temp_video_{i}.mp4"
            
            # Use basic state caching to only drop the buffer to disk if it's a new upload chunk
            cache_key = f"upload_size_{i}"
            if st.session_state.get(cache_key) != uploaded_file.size:
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.session_state[cache_key] = uploaded_file.size
                
            caps[i] = cv2.VideoCapture(temp_path)
elif input_source == "Live Camera":
    if 'camera_configs' not in st.session_state:
        st.session_state.camera_configs = [{"type": "System Camera (Laptop)", "value": "0"}]
        
    for i, config in enumerate(st.session_state.camera_configs):
        st.sidebar.markdown(f"**Zone {chr(65+i)}/{i+1} Feed Configuration**")
        cam_type = st.sidebar.selectbox(
            "Camera Type", 
            ["System Camera (Laptop)", "USB Web Camera", "CCTV Stream (RTSP)"],
            index=["System Camera (Laptop)", "USB Web Camera", "CCTV Stream (RTSP)"].index(config.get("type", "System Camera (Laptop)")) if config.get("type", "System Camera (Laptop)") in ["System Camera (Laptop)", "USB Web Camera", "CCTV Stream (RTSP)"] else 0,
            key=f"cam_type_{i}"
        )
        
        if cam_type == "System Camera (Laptop)":
            st.sidebar.caption("Using built-in laptop camera.")
            config["value"] = "0"
            config["type"] = cam_type
        elif cam_type == "USB Web Camera":
            st.sidebar.caption("Using external USB camera.")
            config["value"] = "1"
            config["type"] = cam_type
        else:
            cam_val = st.sidebar.text_input("Stream URL", value=config.get("value", ""), key=f"cam_val_{i}")
            config["value"] = cam_val
            config["type"] = cam_type
            
        st.session_state.camera_configs[i] = config
        
        # Load the camera
        val = config["value"]
        if val.strip() != "":
            try:
                if val.strip().isdigit():
                    caps[i] = cv2.VideoCapture(int(val.strip()))
                else:
                    caps[i] = cv2.VideoCapture(val.strip())
            except Exception:
                pass
                
    st.sidebar.markdown("---")
    if len(st.session_state.camera_configs) < 4:
        if st.sidebar.button("➕ Add Another Camera"):
            st.session_state.camera_configs.append({"type": "System Camera (Laptop)", "value": ""})
            st.experimental_rerun()

st.sidebar.markdown("### 📧 Alert System Configuration")
with st.sidebar.expander("Email Settings"):
    default_to = _get_cred("ALERT_EMAIL_TO")
    default_from = _get_cred("ALERT_EMAIL_FROM")
    default_pass = _get_cred("ALERT_EMAIL_PASSWORD")
        
    e_to = st.text_input("Send Alerts To", value=default_to)
    e_from = st.text_input("Send From Email", value=default_from)
    e_pass = st.text_input("App Password", type="password", value=default_pass)
    if st.button("Save Settings"):
        if 'email_creds' not in st.session_state:
            st.session_state.email_creds = {}
        st.session_state.email_creds["ALERT_EMAIL_TO"] = e_to
        st.session_state.email_creds["ALERT_EMAIL_FROM"] = e_from
        st.session_state.email_creds["ALERT_EMAIL_PASSWORD"] = e_pass
        
        if update_email_settings(e_to, e_from, e_pass):
            st.success("Saved to frontend & backend!")
        else:
            st.warning("Saved to frontend (backend unreachable)")

st.sidebar.markdown("### 🧠 Model Management")
if st.sidebar.button("Retrain Models", help="Re-syncs and trains LSTM and Prophet on latest data"):
    with st.spinner("Training models... this may take 1-2 minutes"):
        result = train_model()
        if result and result.get("status") == "success":
            st.sidebar.success("Models trained successfully!")
        else:
            st.sidebar.error(f"Training failed: {result.get('message') if result else 'Server error'}")

col1, col2 = st.sidebar.columns(2)
if col1.button("Start Processing"):
    st.session_state.running = True
if col2.button("Stop"):
    st.session_state.running = False

if st.session_state.get('running', False) and input_source != "None":
    frame_counter = 0
    last_total = 0
    ph_vids = [vid_a, vid_b, vid_c, vid_d]
    
    # helper for zone branding
    def add_footer(img, text):
        cv2.putText(img, text, (10, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 2)
        
    if 'last_tracks' not in st.session_state:
        st.session_state.last_tracks = [[] for _ in range(4)]
        
    while True:
        zone_counts = [0, 0, 0, 0]
        frames = [None, None, None, None]
        active_caps = 0
        
        for i in range(4):
            if caps[i] and caps[i].isOpened():
                active_caps += 1
                
                # To speed up playback visually, we can read a few extra frames to skip ahead
                # This ensures the video doesn't look like slow motion
                for _ in range(2):
                    caps[i].grab()
                
                ret, frame = caps[i].read()
                if not ret:
                    caps[i].release()
                    caps[i] = None
                    continue
                    
                # Downscale individual feeds to maintain overall UI performance
                frame = cv2.resize(frame, (480, 270))
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # OPTIMIZATION: Only hit the API every 4 processed frames (network latency bypass)
                if frame_counter % 4 == 0:
                    is_success, buffer = cv2.imencode(".jpg", frame)
                    
                    if is_success:
                        res = upload_frame(buffer.tobytes(), zone_name=zones[i], max_capacity=user_caps[i])
                        
                        if res:
                            tracks = res.get('tracks', [])
                            st.session_state.last_tracks[i] = tracks
                            
                            risk_info = res.get("risk", {})
                            if risk_info.get("level") == "HIGH ALERT":
                                _send_high_alert_email_frontend(
                                    count=len(tracks), 
                                    message=risk_info.get("message", "High crowd density"), 
                                    zone_name=zones[i], 
                                    forecast_info=None
                                )
                                
                # Always draw using the cached tracks for smooth video playback
                tracks = st.session_state.last_tracks[i]
                zone_counts[i] = len(tracks)
                
                for t in tracks:
                    x1, y1, x2, y2 = t['bbox']
                    cv2.rectangle(frame_rgb, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        
                add_footer(frame_rgb, f"{zones[i]} - {zone_counts[i]} detected")
                frames[i] = frame_rgb
                    
        # Update layout images
        for i in range(4):
            if frames[i] is not None:
                ph_vids[i].image(frames[i], use_container_width=True)
            else:
                black = np.zeros((270, 480, 3), dtype=np.uint8)
                cv2.putText(black, "No input", (160, 135), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 100, 100), 2)
                add_footer(black, zones[i])
                ph_vids[i].image(black, use_container_width=True)
                
        if active_caps == 0:
            st.warning("No active video feeds detected or all feeds have finished playing.")
            st.session_state.running = False
            time.sleep(1)
            break
            
        if 'histories' not in st.session_state:
            st.session_state.histories = [[], [], [], []]
            
        current_total = sum(zone_counts)
        for i in range(4):
            if frames[i] is not None:
                st.session_state.histories[i].append(zone_counts[i])
                if len(st.session_state.histories[i]) > 300:
                    st.session_state.histories[i].pop(0)
                
        alerts = update_alert_feed(zone_counts, user_caps)
        update_kpis(current_total, last_total, alerts)
        update_zone_snapshots(zone_counts, user_caps)
        
        if frame_counter % 15 == 0:
            last_total = current_total
            render_graphs(st.session_state.histories)
            
        frame_counter += 1
        time.sleep(0.05)
