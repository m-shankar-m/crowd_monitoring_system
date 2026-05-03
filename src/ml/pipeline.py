import pandas as pd
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
                
                df['ds'] = pd.to_datetime(df['timestamp'])
                df['y'] = df['count'].rolling(window=min(3, len(df)), min_periods=1).mean()
                # New: Add temporal features for inference
                df['hour'] = df['ds'].dt.hour
                df['day'] = df['ds'].dt.dayofweek
                
                # Predictions
                prophet_df = self.prophet_model.predict(periods=periods)
                lstm_preds = self.lstm_model.predict(df, periods=periods)
                
                if prophet_df is None:
                    return []
                    
                results = []
                for i, row in prophet_df.iterrows():
                    p_val = max(0, int(row['yhat']))
                    
                    # Check for ensemble
                    if lstm_preds is not None and i < len(lstm_preds):
                        l_val = max(0, int(lstm_preds[i]))
                        ensem_val = int((p_val + l_val) / 2)
                    else:
                        ensem_val = p_val
                        
                    results.append({
                        "timestamp": row['ds'].strftime("%Y-%m-%d %H:%M:%S"),
                        "predicted_count": ensem_val
                    })
                return results
            except Exception:
                return []
