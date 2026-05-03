from src.ml.pipeline import MLPipeline
from src.risk.alert import predict_future_risk

ml_pipeline = MLPipeline()

def train_system():
    return ml_pipeline.train_model()

def get_predictions(periods=30):
    return ml_pipeline.get_forecast(periods)

def get_predictive_risk(periods=30, history_counts=None):
    forecasts = ml_pipeline.get_forecast(periods, history_counts)
    if not forecasts:
        return {
            "risk": "Unknown", 
            "forecasts": [], 
            "first_high_risk_time": None, 
            "peak_prediction": 0
        }
        
    risk_info = predict_future_risk(forecasts)
    return {
        "forecasts": forecasts,
        "first_high_risk_time": risk_info["first_high_risk_time"],
        "peak_prediction": risk_info["peak_prediction"]
    }
