import pandas as pd
import numpy as np
import os
import threading
from .prophet_model import ForecastModel
from .lstm_model import CrowdLSTMModel

class MLPipeline:
    def __init__(self):
        self.prophet_model = ForecastModel()
        # Updated to match improved defaults
        self.lstm_model = CrowdLSTMModel(sequence_length=20, epochs=50)
        self.csv_path = "data/crowd_data.csv"
        self.models_dir = "models"
        self.lock = threading.Lock()
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Load serialized models dynamically for rapid inference
        p_loaded = self.prophet_model.load()
        l_loaded = self.lstm_model.load()
        
        if not (p_loaded and l_loaded):
            print("WARNING: ML Models not found in 'models/' directory. Please execute 'notebooks/train_models.ipynb' first!")

    def train_model(self):
        with self.lock:
            if not os.path.exists(self.csv_path):
                return {"status": "error", "message": f"Data file {self.csv_path} not found"}
            
        try:
            df = pd.read_csv(self.csv_path)
            
            # Data cleaning: Ensure numeric and drop NaNs
            df['count'] = pd.to_numeric(df['count'], errors='coerce')
            df = df.dropna(subset=['count', 'timestamp'])
            
            if len(df) < 25: # Increased to ensure enough for sequence_length=20
                return {"status": "error", "message": f"Not enough data to train models (need 25+ rows, found {len(df)})"}
                
            df['ds'] = pd.to_datetime(df['timestamp'])
            df['y'] = df['count'].rolling(window=3, min_periods=1).mean()
            # New: Add temporal features
            df['hour'] = df['ds'].dt.hour
            df['day'] = df['ds'].dt.dayofweek
            
            # Using only the most recent data to train quickly
            df_train = df.tail(5000).copy()
            
            # Train Prophet
            prophet_success = self.prophet_model.train(df_train[['ds', 'y', 'hour', 'day']])
            if prophet_success:
                self.prophet_model.save()
            
            # Train LSTM
            lstm_success = self.lstm_model.train(df_train[['ds', 'y', 'hour', 'day']])
            if lstm_success:
                self.lstm_model.save()
            
            if prophet_success and lstm_success:
                return {"status": "success", "message": "Ensemble (Prophet + LSTM) trained successfully"}
            elif prophet_success:
                return {"status": "success", "message": "Only Prophet trained successfully"}
            elif lstm_success:
                return {"status": "success", "message": "Only LSTM trained successfully"}
                
            return {"status": "error", "message": "Ensemble training failed completely"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def get_forecast(self, periods=30, history_counts=None):
        with self.lock:
            try:
                if history_counts is not None and len(history_counts) > 0:
                    import time
                    current_time = time.time()
                    df = pd.DataFrame({
                        'timestamp': [time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(current_time - (len(history_counts)-i)*2)) for i in range(len(history_counts))],
                        'count': history_counts
                    })
                else:
                    df = pd.read_csv(self.csv_path)
                    
                if len(df) < 2:
                    return []
                
                df['ds'] = pd.to_datetime(df['timestamp'])
                df['y'] = df['count'].rolling(window=min(3, len(df)), min_periods=1).mean()
                # New: Add temporal features for inference
                df['hour'] = df['ds'].dt.hour
                df['day'] = df['ds'].dt.dayofweek
                
                # Predictions
                prophet_df = self.prophet_model.predict(periods=periods)
                lstm_preds = self.lstm_model.predict(df, periods=periods)
                
                results = []
                last_val = history_counts[-1] if history_counts is not None and len(history_counts) > 0 else 0
                
                # Calculate real-time trend if we have enough data
                trend = 0
                if history_counts is not None and len(history_counts) > 2:
                    recent_diffs = np.diff(history_counts[-5:]) if len(history_counts) >= 5 else np.diff(history_counts)
                    trend = np.mean(recent_diffs)
                    # Dampen the trend to prevent crazy extrapolation
                    trend = np.clip(trend, -2.0, 2.0)
                
                # Prophet dataframe might have non-zero starting index, so we reset it
                if prophet_df is not None:
                    prophet_df = prophet_df.reset_index(drop=True)
                    
                for idx in range(periods):
                    # 1. Base Naive Extrapolation
                    naive_pred = max(0, last_val + (trend * (idx + 1)))
                    
                    # 2. Get AI Prediction
                    ai_pred = None
                    if lstm_preds is not None and idx < len(lstm_preds):
                        ai_pred = max(0, lstm_preds[idx])
                    elif prophet_df is not None and idx < len(prophet_df):
                        ai_pred = max(0, prophet_df.iloc[idx]['yhat'])
                        
                    # 3. Smart Blending
                    if ai_pred is not None and ai_pred > 0:
                        # Smoothly transition from live naive trend to AI baseline
                        # idx=0 means 90% naive, idx=14 means 60% AI
                        alpha = min(0.6, (idx + 1) / periods)
                        pred_val = int(naive_pred * (1 - alpha) + ai_pred * alpha)
                    else:
                        pred_val = int(naive_pred)
                        
                    # Calculate future timestamp based on last input time + 3s steps
                    future_ts = df['ds'].iloc[-1] + pd.Timedelta(seconds=3 * (idx + 1))
                    
                    results.append({
                        "timestamp": future_ts.strftime("%Y-%m-%d %H:%M:%S"),
                        "predicted_count": pred_val
                    })
                return results
            except Exception:
                return []
