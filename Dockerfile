FROM python:3.11-slim

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face requires running as a non-root user
RUN useradd -m -u 1000 user
USER user

ENV PATH="/home/user/.local/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy the application with correct permissions
COPY --chown=user . /app

# Move to the actual project folder
WORKDIR /app/crowd_monitoring_system

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download YOLO weights to avoid download delay at startup
RUN python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# Ensure directories exist for YOLO weights and logs
RUN mkdir -p logs data/models data/temp_videos

# Hugging Face Spaces uses port 7860
EXPOSE 7860

# Start FastAPI backend
CMD uvicorn src.backend.main:app --host 0.0.0.0 --port ${PORT:-7860}
