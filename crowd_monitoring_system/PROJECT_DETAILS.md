# Crowd Monitoring System - Project Documentation

This document provides a detailed technical explanation of the codebase, the technologies used, and the data requirements for each component of the Crowd Monitoring System.

---

## 🏗️ Architecture Overview

The system follows a modular architecture consisting of five main layers:
1.  **Computer Vision (CV) Layer**: Handles real-time person/head detection (YOLO) and density estimation (CSRNet).
2.  **Machine Learning (ML) Layer**: Responsible for time-series forecasting of crowd density trends.
3.  **Backend API**: A FastAPI service that orchestrates CV processing, ML training/inference, and alerting.
4.  **Risk & Alerting**: A logic layer that evaluates density levels against dynamic thresholds and triggers notifications.
5.  **Frontend Dashboard**: A Streamlit-based interactive UI for real-time monitoring and analytics.

---

## 🛠️ Technology Stack

| Component | Technologies |
| :--- | :--- |
| **Core Language** | Python 3.10+ |
| **Frontend** | Streamlit, Plotly, CSS (Glassmorphism) |
| **Backend API** | FastAPI, Uvicorn |
| **Computer Vision** | OpenCV, Ultralytics (YOLOv8), CSRNet (PyTorch) |
| **Machine Learning** | PyTorch (LSTM), Facebook Prophet, Pandas, NumPy |
| **Environment** | Python-dotenv, SMTP (SSL/TLS) |

---

## 📁 Detailed File Explanations

### 1. Computer Vision Layer (`src/cv/`)

- **`detector.py`**: 
    - **Purpose**: Implements the `PersonDetector` class.
    - **Technology**: Uses **YOLOv8** (specifically `yolov8m.pt`).
    - **Optimization**: Configured to prioritize **head detection** to minimize occlusion issues in dense crowds. Uses `imgsz=1024` and `conf=0.15` for fine-grained detection.
- **`density_estimator.py`**:
    - **Purpose**: Implements **CSRNet** for density map generation.
    - **Technology**: **PyTorch**.
    - **Logic**: Used for estimating counts in highly congested areas where individual tracking fails.
- **`tracker.py`**:
    - **Purpose**: Tracks individuals across frames to maintain unique IDs and accurate counts.
- **`pipeline.py`**:
    - **Purpose**: Orchestrates the hybrid YOLO/CSRNet detection and tracking into a single processing pipeline.

### 2. Machine Learning Layer (`src/ml/`)

- **`lstm_model.py`**:
    - **Purpose**: Deep learning model for short-term crowd forecasting.
    - **Technology**: **PyTorch**.
    - **Logic**: Implements a many-to-one **LSTM** network for temporal prediction.
- **`prophet_model.py`**:
    - **Purpose**: Time-series forecasting using statistical methods.
    - **Technology**: **Facebook Prophet**.
    - **Logic**: Captures daily/weekly trends and seasonal patterns.
- **`train_combined_models.py`**:
    - **Purpose**: Utility to train both Prophet and LSTM models.

### 3. Backend API (`src/backend/`)

- **`main.py`**:
    - **Purpose**: Entry point for the FastAPI server.
    - **Endpoints**:
        - `POST /live-density`: Processes video frames, returns counts, risk levels, and bounding boxes.
        - `POST /train`: Triggers model retraining.
        - `GET /predict`: Returns forecasted crowd trends.
        - `GET /test-email-alert`: Utility to verify SMTP settings.

### 4. Risk & Alerting Layer (`src/risk/`)

- **`threshold.py`**:
    - **Purpose**: Defines dynamic logic for crowd density categorization.
    - **Logic**: Reads `MAX_CAPACITY` from environment variables and calculates LOW/MODERATE/HIGH boundaries dynamically.
- **`alert.py`**:
    - **Purpose**: Handles notifications for threshold breaches.
    - **Features**:
        - **Multi-Zone Routing**: Supports 4 distinct zones with specific alert messages.
        - **SMTP Integration**: Sends secure email alerts.
        - **Cooldown Mechanism**: Prevents repeated alerts (configurable via `.env`).

### 5. Frontend Dashboard (`src/frontend/`)

- **`app.py`**:
    - **Purpose**: Main user interface built with **Streamlit**.
    - **Features**:
        - **Glassmorphism UI**: Modern, dark-themed layout.
        - **4-Zone Monitoring**: Real-time stats for 4 different areas.
        - **Dynamic Configuration**: UI controls to adjust thresholds on the fly.
        - **Visual Analytics**: Plotly charts for actual vs. predicted trends.
- **`api.py`**: Communication layer between Frontend and Backend.

---

## 📊 Data Requirements

### 1. Inputs
- **Video Streams**: MP4, AVI, or local webcam.
- **Historical Data**: CSV/JSON files for training.

### 2. Configuration (`.env`)
- `ALERT_EMAIL_TO`, `ALERT_EMAIL_FROM`, `ALERT_EMAIL_PASSWORD`: SMTP settings.
- `MAX_CAPACITY`: The global crowd limit.
- `ALERT_EMAIL_COOLDOWN_SECONDS`: Cooldown period.

---

## 🚀 How it Works (Data Flow)

1.  **Frontend** captures a frame and sends it to the **Backend**.
2.  **Backend** runs the **Hybrid CV Pipeline** (YOLO for sparse, CSRNet for dense).
3.  **Risk Layer** evaluates the count against dynamic thresholds.
4.  **Alerting System** triggers notifications if limits are exceeded, respecting zone-specific logic and cooldowns.
5.  **ML Forecasting** runs in parallel or on-demand to predict future trends.
6.  **Frontend** updates the dashboard with live telemetry, alerts, and predictions.
