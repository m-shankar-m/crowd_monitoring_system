# Crowd Monitoring & Prediction System

A robust, real-time computer vision and machine learning platform designed to monitor crowd density from live video feeds, identify immediate safety thresholds, and forecast future density trends to preemptively highlight high-risk situations.

**GitHub Repository:** [https://github.com/m-shankar-m/crowd_monitoring_system.git](https://github.com/m-shankar-m/crowd_monitoring_system.git)

## 🚀 Key Features

*   **Hybrid Crowd Analysis:** Combines `YOLOv8` for precise person/head tracking in sparse environments and `CSRNet` (Congested Scene Recognition Network) for accurate density estimation in highly congested areas.
*   **Optimized Head Detection:** Utilizes a custom-tuned YOLOv8 model focused on head detection to improve accuracy and reduce occlusion issues in dense crowds.
*   **Multi-Zone Monitoring:** Supports simultaneous monitoring of up to 4 distinct zones (Zone A, Zone B, Zone C, Zone D) with zone-specific alert logic.
*   **Predictive Alert Outlook:** High-risk email alerts now include a 'Forecast Outlook', predicting peak crowd sizes and risk duration to aid in rapid staff deployment.
*   **High-Accuracy ML Engine:** Features an upgraded multi-variate LSTM achieving **~85% accuracy** by analyzing temporal patterns (hour of day, day of week).
*   **Dynamic Threshold Configuration:** Allows real-time adjustment of crowd capacity limits via the frontend dashboard.
*   **Interactive Analytics Dashboard:** A comprehensive Streamlit interface presenting live video rendering, real-time numeric KPIs, and dynamic actual-vs-predicted trajectory graphs.

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
├── data/                       # Operational history CSVs and samples
│   ├── samples/                # Sample video feeds
│   ├── processed/              # Filtered datasets
│   └── raw/                    # Original telemetry files
├── docs/                       # Extended documentation and project details
├── models/                     # YOLO and ML weights
├── notebooks/                  # Interactive training environments
├── scripts/                    # Maintenance and generation utilities
├── src/
│   ├── backend/                # REST API logic
│   ├── cv/                     # Computer Vision pipeline
│   ├── frontend/               # Streamlit UI dashboard
│   ├── ml/                     # ML training classes
│   └── risk/                   # Alerting and threshold logic
├── .env                        # Environment configuration
└── README.md                   # System overview
```

