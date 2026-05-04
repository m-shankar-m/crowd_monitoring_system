from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from src.backend.services.density import process_image
from src.backend.services.forecast import train_system, get_predictions, get_predictive_risk
from src.risk.alert import generate_alert

app = FastAPI(title="Crowd_Predictor_Framework API")

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"status": "alive", "service": "Crowd Monitoring API"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/live-density")
def live_density(file: UploadFile = File(...), zone_name: str = "Unknown", max_capacity: int = 25):
    try:
        import shutil
        import tempfile
        import os
        import cv2
        from fastapi.responses import StreamingResponse
        import json
        
        filename = file.filename.lower() if file.filename else ""
        is_video = filename.endswith((".mp4", ".avi", ".mov", ".mkv")) or file.content_type.startswith("video")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4" if is_video else ".jpg") as tf:
            shutil.copyfileobj(file.file, tf)
            tf_path = tf.name
            
        from src.backend.services.density import cv_pipeline
        
        if is_video:
            def video_generator():
                try:
                    for res in cv_pipeline.process_video_stream(tf_path):
                        yield json.dumps(res) + "\n"
                finally:
                    if os.path.exists(tf_path):
                        os.remove(tf_path)
            return StreamingResponse(video_generator(), media_type="application/x-ndjson")
        else:
            try:
                frame = cv2.imread(tf_path)
                if frame is None:
                    raise ValueError("Invalid image")
                results = cv_pipeline.process_frame(frame)
                
                # Predict risk if needed
                forecast_info = None
                if results["count"] > 0.5 * max_capacity:
                    forecast_info = get_predictive_risk(periods=15)
                    
                alert_info = generate_alert(results["count"], zone_name, max_capacity, forecast_info)
                results["risk"] = alert_info
                return results
            finally:
                if os.path.exists(tf_path):
                    os.remove(tf_path)
                # Cleanup single frame memory
                import gc; gc.collect()
                import torch; 
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                
    except Exception as e:
        print(f"ERROR processing live-density: {str(e)}")
        return {
            "count": 0,
            "tracks": [],
            "error": "Backend Error",
            "detail": str(e)
        }

@app.post("/train")
def train():
    return train_system()

from pydantic import BaseModel
from typing import List, Optional

class EmailSettings(BaseModel):
    email_to: str
    email_from: str
    email_password: str

@app.post("/update-email-settings")
def update_email_settings(settings: EmailSettings):
    import dotenv
    import os
    env_path = ".env"
    if not os.path.exists(env_path):
        open(env_path, 'w').close()
    dotenv.set_key(env_path, "ALERT_EMAIL_TO", settings.email_to)
    dotenv.set_key(env_path, "ALERT_EMAIL_FROM", settings.email_from)
    dotenv.set_key(env_path, "ALERT_EMAIL_PASSWORD", settings.email_password)
    
    os.environ["ALERT_EMAIL_TO"] = settings.email_to
    os.environ["ALERT_EMAIL_FROM"] = settings.email_from
    os.environ["ALERT_EMAIL_PASSWORD"] = settings.email_password
    return {"status": "success"}

class PredictRequest(BaseModel):
    periods: int = 30
    history_counts: Optional[List[int]] = None

@app.post("/predict-zone")
def predict_zone(req: PredictRequest):
    return get_predictive_risk(req.periods, req.history_counts)

@app.get("/predict")
def predict(periods: int = 30):
    return get_predictions(periods)

@app.get("/predict-risk")
def predict_risk(periods: int = 30):
    return get_predictive_risk(periods)

@app.get("/test-email-alert")
def test_email_alert(count: int = 30, zone_name: str = "Test Zone", max_capacity: int = 25):
    # Useful for verifying SMTP setup quickly without video upload.
    return generate_alert(count, zone_name, max_capacity)
