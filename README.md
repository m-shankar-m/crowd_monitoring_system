---
title: Crowd Monitoring Backend
emoji: 🏢
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---
# Crowd Monitoring & Prediction System

A robust, real-time computer vision and machine learning platform designed to monitor crowd density from live video feeds, identify immediate safety thresholds, and forecast future density trends to preemptively highlight high-risk situations.

**GitHub Repository:** [https://github.com/m-shankar-m/crowd_monitoring_system.git](https://github.com/m-shankar-m/crowd_monitoring_system.git)

## 🚀 Key Features

*   **Hybrid Crowd Analysis:** Combines `YOLOv8` for precise person/head tracking in sparse environments and `CSRNet` (Congested Scene Recognition Network) for accurate density estimation in highly congested areas.
*   **Optimized Head Detection:** Utilizes a custom-tuned YOLOv8 model focused on head detection to improve accuracy and reduce occlusion issues in dense crowds.
*   **Predictive Forecasting:** Incorporates machine learning time-series models (`Prophet` & `LSTM`) to predict near-future crowd spikes based on recent historical accumulation data.
*   **Multi-Zone Monitoring:** Supports simultaneous monitoring of up to 4 distinct zones (Zone 1, Zone 2, Zone 3, Zone 4) with zone-specific alert logic.
*   **Dynamic Threshold Configuration:** Allows real-time adjustment of crowd capacity limits (Maximum Capacity, Warning Thresholds) via the `.env` file or the frontend dashboard.
*   **Interactive Analytics Dashboard:** A comprehensive Streamlit interface presenting live video rendering, real-time numeric KPIs, dynamic actual-vs-predicted trajectory graphs, and 2D spatial density heatmaps.

## 🛠️ Technology Stack

*   **Computer Vision Framework:** OpenCV, Ultralytics YOLO (`yolov8m.pt`), CSRNet (PyTorch)
*   **Machine Learning / Data Processing:** Facebook Prophet, TensorFlow LSTM, Pandas, Plotly Express
*   **Application Backend:** FastAPI, Uvicorn
*   **Frontend User Interface:** Streamlit

## ⚙️ How to Run the System

This project is separated into a FastAPI backend engine (handling AI inference) and a Streamlit frontend (displaying calculations seamlessly). **Both services must be running simultaneously.**

### 1. Start the Backend API (Computer Vision & ML Logic)

Open a new terminal, ensure your virtual environment is active, and launch the REST API server:

```bash
cd crowd_monitoring_system
python -m uvicorn src.backend.main:app --port 8000
```
*(Wait until you see `Application startup complete.`)*

### 2. Start the Frontend Dashboard (User Interface)

Open a **second separate terminal**, ensure your environment is active, and launch the dashboard:

```bash
cd crowd_monitoring_system
streamlit run src/frontend/app.py
```

### 3. Usage inside the Browser

1. Localhost should automatically open in your web browser (typically `http://localhost:8501`).
2. **Video Upload Logic**: On the left-hand panel under "Live Feed", directly drag-and-drop or browse for `.mp4`, `.avi`, or `.mkv` files simulating security camera footage.
3. The dashboard will instantly process frames, generating bounding tracking elements and updating the UI metrics & predictive graphs every few seconds depending on framerate complexity.

### 4. Alert Configuration

The system uses a flexible alerting mechanism. Set these environment variables in your `.env` file:

```bash
# Email Credentials
ALERT_EMAIL_TO=recipient@example.com
ALERT_EMAIL_FROM=sender@example.com
ALERT_EMAIL_PASSWORD=your_app_password
ALERT_SMTP_HOST=smtp.gmail.com
ALERT_SMTP_PORT=465

# Thresholds
MAX_CAPACITY=50
ALERT_EMAIL_COOLDOWN_SECONDS=60
```

Notes:
- `MAX_CAPACITY` defines the global limit before a critical alert is triggered.
- Multi-zone alerts are automatically routed to the configured email with specific zone information.
- Cooldown prevents repeated emails every frame while crowd remains high.

## 📁 Project Architecture

```plaintext
crowd_monitoring_system/
├── data/                       # Operational history data CSVs used for prediction
├── models/                     # Pre-trained YOLO and CSRNet weights
├── src/
│   ├── backend/                # REST API (main.py pipeline triggers)
│   ├── cv/                     # YOLO/CSRNet integration and image filters
│   ├── frontend/               # Streamlit application visual engine (app.py)
│   ├── ml/                     # Prophet/LSTM time-series data fitting
│   └── risk/                   # Conditional scaling logic (threshold.py, alert.py)
└── README.md                   # You are here!
```

