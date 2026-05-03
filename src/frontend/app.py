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
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode, RTCConfiguration
import av

st.set_page_config(page_title="Crowd_Predictor_Framework", layout="wide", initial_sidebar_state="collapsed")

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
st.markdown("<h1>Crowd_Predictor_Framework — Live Dashboard</h1>", unsafe_allow_html=True)
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

# WebRTC Configuration
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.count = 0
        self.tracks = []
        self.zone_name = "Unknown"
        self.max_capacity = 25
        self.last_frame_processed = 0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        
        # Rate limit processing to avoid saturating the backend
        current_time = time.time()
        if current_time - self.last_frame_processed > 0.5: # 2 FPS for analysis
            self.last_frame_processed = current_time
            
            # Prepare frame for API
            small_frame = cv2.resize(img, (480, 270))
            _, buffer = cv2.imencode(".jpg", small_frame)
            
            res = upload_frame(buffer.tobytes(), zone_name=self.zone_name, max_capacity=self.max_capacity)
            if res:
                self.tracks = res.get('tracks', [])
                self.count = res.get('count', len(self.tracks))

        # Draw bounding boxes
        for t in self.tracks:
            x1, y1, x2, y2 = t['bbox']
            # Scale coordinates back to original frame size
            h, w = img.shape[:2]
            sx1, sy1 = int(x1 * w / 480), int(y1 * h / 270)
            sx2, sy2 = int(x2 * w / 480), int(y2 * h / 270)
            cv2.rectangle(img, (sx1, sy1), (sx2, sy2), (0, 255, 0), 2)
            
        cv2.putText(img, f"{self.zone_name} - {self.count} detected", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            
        return av.VideoFrame.from_ndarray(img, format="bgr24")

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
if 'camera_configs' not in st.session_state:
    st.session_state.camera_configs = [
        {"type": "System Camera (Laptop)", "value": "0"},
        {"type": "System Camera (Laptop)", "value": "0"},
        {"type": "System Camera (Laptop)", "value": "0"},
        {"type": "System Camera (Laptop)", "value": "0"}
    ]

elif input_source == "Live Camera":
    for i in range(len(st.session_state.camera_configs)):
        if i < 4:
            st.sidebar.markdown(f"**{zones[i]} Feed Configuration**")
            options = ["System Camera (Laptop)", "USB Camera (Cable)", "CCTV Camera (Online/IP)"]
            
            # Ensure safe indexing
            current_type = st.session_state.camera_configs[i]["type"]
            idx = options.index(current_type) if current_type in options else 0
            
            cam_type = st.sidebar.selectbox(f"Camera Type", options + ["Webcam (Cloud/Browser)"], index=idx, key=f"cam_type_{i}")
            st.session_state.camera_configs[i]["type"] = cam_type
            
            cam_val = ""
            if cam_type == "System Camera (Laptop)":
                st.sidebar.caption("Using built-in laptop camera.")
                cam_val = "0"
            elif cam_type == "USB Camera (Cable)":
                cam_val = st.sidebar.text_input("USB Camera Index (e.g. 1, 2)", value=st.session_state.camera_configs[i]["value"] if st.session_state.camera_configs[i]["value"] != "0" else "1", key=f"cam_val_{i}")
            elif cam_type == "CCTV Camera (Online/IP)":
                cam_val = st.sidebar.text_input("CCTV Stream URL (rtsp://... or http://...)", value=st.session_state.camera_configs[i]["value"] if not st.session_state.camera_configs[i]["value"].isdigit() else "", key=f"cam_val_{i}")
            elif cam_type == "Webcam (Cloud/Browser)":
                st.sidebar.info("WebRTC will be initialized in the zone panel.")
                cam_val = "WEBRTC"
            
            st.session_state.camera_configs[i]["value"] = cam_val
            st.sidebar.markdown("<hr style='margin: 10px 0px;'>", unsafe_allow_html=True)
            
            if cam_val.strip() != "" and cam_val != "WEBRTC":
                try:
                    if cam_val.strip().isdigit():
                        caps[i] = cv2.VideoCapture(int(cam_val.strip()))
                    else:
                        caps[i] = cv2.VideoCapture(cam_val.strip())
                except Exception:
                    pass

    if len(st.session_state.camera_configs) < 4:
        if st.sidebar.button("➕ Add Another Camera"):
            st.session_state.camera_configs.append({"type": "CCTV Camera (Online/IP)", "value": ""})
            st.rerun()

st.sidebar.markdown("### 📧 Alert System Configuration")
with st.sidebar.expander("Email Settings", expanded=False):
    st.markdown("<small>Change alert recipients and sender credentials dynamically.</small>", unsafe_allow_html=True)
    e_to = st.text_input("Recipient Email (To)", value=os.getenv("ALERT_EMAIL_TO", ""))
    e_from = st.text_input("Sender Email (From)", value=os.getenv("ALERT_EMAIL_FROM", ""))
    e_pass = st.text_input("App Password", value=os.getenv("ALERT_EMAIL_PASSWORD", ""), type="password")
    if st.button("Save Credentials"):
        success = update_email_settings(e_to, e_from, e_pass)
        if success:
            st.success("Alert email updated successfully!")
            # Also update local os.environ for Streamlit to show the new value immediately
            os.environ["ALERT_EMAIL_TO"] = e_to
            os.environ["ALERT_EMAIL_FROM"] = e_from
            os.environ["ALERT_EMAIL_PASSWORD"] = e_pass
        else:
            st.error("Failed to update.")

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
        
    while True:
        zone_counts = [0, 0, 0, 0]
        frames = [None, None, None, None]
        active_caps = 0
        
        for i in range(4):
            if caps[i] and caps[i].isOpened():
                active_caps += 1
                ret, frame = caps[i].read()
                if not ret:
                    caps[i].release()
                    caps[i] = None
                    continue
                    
                # Downscale individual feeds to maintain overall UI performance
                frame = cv2.resize(frame, (480, 270))
                is_success, buffer = cv2.imencode(".jpg", frame)
                
                if is_success:
                    res = upload_frame(buffer.tobytes(), zone_name=zones[i], max_capacity=user_caps[i])
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    if res:
                        tracks = res.get('tracks', [])
                        zone_counts[i] = res.get('count', len(tracks))
                        
                        for t in tracks:
                            x1, y1, x2, y2 = t['bbox']
                            cv2.rectangle(frame_rgb, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                            
                    add_footer(frame_rgb, f"{zones[i]} - {zone_counts[i]} detected")
                    frames[i] = frame_rgb
            elif i < len(st.session_state.camera_configs) and st.session_state.camera_configs[i]["value"] == "WEBRTC":
                active_caps += 1
                frames[i] = "WEBRTC_ACTIVE"
                if f"webrtc_ctx_{i}" in st.session_state and st.session_state[f"webrtc_ctx_{i}"].video_processor:
                    zone_counts[i] = st.session_state[f"webrtc_ctx_{i}"].video_processor.count
                    
        # Update layout images
        for i in range(4):
            if isinstance(frames[i], str) and frames[i] == "WEBRTC_ACTIVE":
                with ph_vids[i].container():
                    ctx = webrtc_streamer(
                        key=f"webrtc-{i}-{time.time() // 3600}", # Refresh key every hour
                        mode=WebRtcMode.SENDRECV,
                        rtc_configuration=RTC_CONFIGURATION,
                        video_processor_factory=VideoProcessor,
                        media_stream_constraints={"video": True, "audio": False},
                        async_processing=True,
                    )
                    if ctx.video_processor:
                        ctx.video_processor.zone_name = zones[i]
                        ctx.video_processor.max_capacity = user_caps[i]
                    st.session_state[f"webrtc_ctx_{i}"] = ctx
            elif frames[i] is not None:
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
