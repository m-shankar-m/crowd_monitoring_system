import streamlit as st
import sys
import os
import traceback
import time
import random
import threading

# --- GLOBAL STARTUP WRAPPER ---
try:
    import cv2
    import numpy as np
    import plotly.graph_objects as go
    import pandas as pd
    from PIL import Image
    from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

    # Add project root to sys.path
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # Import direct backend services
    try:
        from src.backend.services.density import process_image
        from src.risk.alert import generate_alert
        from src.backend.services.forecast import get_predictive_risk
    except ImportError:
        # Fallback if pathing is different on cloud
        sys.path.append(os.getcwd())
        from src.backend.services.density import process_image
        from src.risk.alert import generate_alert
        from src.backend.services.forecast import get_predictive_risk

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

    # UI Configuration
    st.set_page_config(page_title="Crowd Monitoring System", layout="wide", initial_sidebar_state="expanded")

    st.title("🏙️ Crowd Monitoring Dashboard")
    
    # Sidebar
    st.sidebar.header("Camera Configuration")
    input_source = st.sidebar.selectbox("Select Source", ["Browser Camera (WebRTC)", "None"])
    
    if input_source == "Browser Camera (WebRTC)":
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

    # Status info
    st.sidebar.markdown("---")
    st.sidebar.subheader("System Status")
    st.sidebar.write(f"Inference: {'Active' if st.session_state.get('running') else 'Stopped'}")
    st.sidebar.write(f"Frames Received: {st.session_state.webrtc_count}")
    if DEBUG_INFO["error"]:
        st.sidebar.error(DEBUG_INFO["error"])

    # Main Dashboard
    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("Live Feed")
        video_ph = st.empty()
    with c2:
        st.subheader("Real-time Stats")
        metric_ph = st.empty()
        
    # Processing Loop
    if st.session_state.get("running"):
        video_ph.info("Processing started...")
        while st.session_state.get("running"):
            frame = None
            with st.session_state.webrtc_lock:
                if 0 in st.session_state.webrtc_frames:
                    frame = st.session_state.webrtc_frames[0].copy()
            
            if frame is not None:
                is_success, buffer = cv2.imencode(".jpg", frame)
                if is_success:
                    results = get_analysis_direct(buffer.tobytes(), "Main Zone", 100)
                    if results:
                        video_ph.image(frame, channels="BGR")
                        metric_ph.metric("People Detected", results["count"])
            
            time.sleep(0.5)

except Exception as e:
    st.error(f"🚀 Startup Failure: {e}")
    st.code(traceback.format_exc())
