import pandas as pd
import numpy as np
import os
import torch
import pickle
from prophet.serialize import model_to_json

import sys

# Import local models
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.ml.lstm_model import CrowdLSTMModel
from src.ml.prophet_model import ForecastModel

def load_data():
    df_list = []
    
    # 1. Primary Live Dataset
    try:
        if os.path.exists('data/crowd_data.csv'):
            df_live = pd.read_csv('data/crowd_data.csv')
            # Filter capped and zero values
            df_live = df_live[(df_live['count'] != 500.0) & (df_live['count'] > 0)]
            df_live['timestamp'] = pd.to_datetime(df_live['timestamp'])
            # Ensure hour/day exist
            if 'hour' not in df_live.columns:
                df_live['hour'] = df_live['timestamp'].dt.hour
                df_live['day'] = df_live['timestamp'].dt.dayofweek
            df_list.append(df_live[['timestamp', 'count', 'hour', 'day']])
    except Exception as e:
        print("Could not parse data/crowd_data.csv:", e)

    # 2. Historical Raw Dataset
    try:
        if os.path.exists('data/raw/crowd_data.csv'):
            df1 = pd.read_csv('data/raw/crowd_data.csv')
            df1 = df1[(df1['count'] != 500.0) & (df1['count'] > 0)]
            df1['timestamp'] = pd.to_datetime(df1['timestamp'])
            if 'hour' not in df1.columns:
                df1['hour'] = df1['timestamp'].dt.hour
                df1['day'] = df1['timestamp'].dt.dayofweek
            df_list.append(df1[['timestamp', 'count', 'hour', 'day']])
    except Exception as e:
        print("Could not parse data/raw/crowd_data.csv:", e)
        
    # 2. crowd_counts.csv (Processed tracker context)
    try:
        df2 = pd.read_csv('data/processed/crowd_counts.csv', names=['timestamp', 'count'])
        df2['timestamp'] = pd.to_datetime(df2['timestamp'], errors='coerce')
        df2 = df2.dropna(subset=['timestamp'])
        df2['hour'] = df2['timestamp'].dt.hour
        df2['day'] = df2['timestamp'].dt.dayofweek
        df_list.append(df2[['timestamp', 'count', 'hour', 'day']])
    except Exception as e:
        print("Could not parse crowd_counts.csv:", e)
        
    # 3. pedestrian_data.csv (Synthesized timeline starting Feb '24)
    try:
        df3 = pd.read_csv('data/raw/pedestrian_data.csv')
        df3['count'] = df3['Crowd Count']
        # Fix: Apply same filter for pedestrian data
        df3 = df3[(df3['count'] != 500.0) & (df3['count'] > 0)]
        start_date = pd.to_datetime('2024-02-01 00:00:00')
        df3['timestamp'] = start_date + pd.to_timedelta(df3.index * 5, unit='min')
        df3['hour'] = df3['timestamp'].dt.hour
        df3['day'] = df3['timestamp'].dt.dayofweek
        df_list.append(df3[['timestamp', 'count', 'hour', 'day']])
    except Exception as e:
        print("Could not parse pedestrian_data.csv:", e)
        
    # 4. shanghaitech_data.csv (Synthesized timeline starting Mar '24)
    try:
        df4 = pd.read_csv('data/raw/shanghaitech_data.csv')
        # Fix: Properly parse timedelta timestamps and anchor to real date
        df4['offset_min'] = pd.to_timedelta(df4['Time']).dt.total_seconds() / 60
        grouped = df4.groupby('offset_min')['Count'].mean().reset_index()
        base = pd.Timestamp('2024-03-01 00:00:00')
        grouped['timestamp'] = base + pd.to_timedelta(grouped['offset_min'], unit='min')
        grouped['count'] = grouped['Count']
        grouped['hour'] = grouped['timestamp'].dt.hour
        grouped['day'] = grouped['timestamp'].dt.dayofweek
        df_list.append(grouped[['timestamp', 'count', 'hour', 'day']])
    except Exception as e:
        print("Could not parse shanghaitech_data.csv:", e)
        
    combined = pd.concat(df_list, ignore_index=True)
    combined = combined.sort_values('timestamp').reset_index(drop=True)
    
    # Fill any null constraints and format
    combined['count'] = combined['count'].ffill()
    combined['ds'] = combined['timestamp']
    combined['y'] = combined['count'].rolling(window=3, min_periods=1).mean()
    
    return combined

if __name__ == '__main__':
    print("Loading and consolidating datasets...")
    df = load_data()
    print(f"Total historical datapoints collected: {len(df)}")
    
    # Cap size to avoid extreme training times on large combined sets
    # We take the most recent 10000 points to ensure responsive training
    df_train = df.tail(10000).copy()
    print(f"Training on the trailing {len(df_train)} datapoints.")
    
    os.makedirs('models', exist_ok=True)
    
    # 1. Train Prophet
    print("\n--- Training Prophet Forecast Model ---")
    prophet = ForecastModel()
    prophet.train(df_train[['ds', 'y', 'hour', 'day']])
    with open('models/prophet_model.json', 'w') as f:
        f.write(model_to_json(prophet.model))
    print("[OK] Prophet model saved to models/prophet_model.json")
    
    # 2. Train LSTM
    print("\n--- Training LSTM High-Frequency Model ---")
    # Updated hyperparameters for better accuracy (Lower LR for stability)
    lstm = CrowdLSTMModel(sequence_length=20, epochs=50, lr=0.001)
    lstm.train(df_train[['ds', 'y', 'hour', 'day']])
    torch.save(lstm.model.state_dict(), 'models/lstm_weights.pth')
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(lstm.scaler, f)
    print("[OK] LSTM weights and scaler saved to models/")
    
    # 3. Evaluation (Simple sanity check on trailing data)
    print("\n--- Model Evaluation (Training Set Sanity Check) ---")
    try:
        # Prophet evaluation (In-sample comparison)
        # We need to ensure we pass the regressors
        p_eval_df = df_train[['ds', 'hour', 'day']].copy()
        p_forecast_full = prophet.model.predict(p_eval_df)
        p_mae = np.abs(p_forecast_full['yhat'].values - df_train['y'].values).mean()
        p_acc = 100 * (1 - p_mae / df_train['y'].mean())
        print(f"Prophet - MAE: {p_mae:.2f}, Approximate Accuracy: {p_acc:.2f}%")
        
        # LSTM evaluation (1-step ahead on last 100 points)
        lstm.model.eval()
        features = ['y', 'hour', 'day']
        test_data = df_train[features].tail(120).values
        test_data_norm = lstm.scaler.transform(test_data)
        
        preds = []
        with torch.no_grad():
            for i in range(100):
                seq = torch.FloatTensor(test_data_norm[i:i+20]).unsqueeze(0)
                pred = lstm.model(seq).item()
                preds.append(pred)
        
        # Inverse transform only the count column (pad with zeros for hour/day)
        preds_padded = np.zeros((len(preds), 3))
        preds_padded[:, 0] = preds
        preds_orig = lstm.scaler.inverse_transform(preds_padded)[:, 0]
        
        actuals = test_data[20:, 0].flatten()
        l_mae = np.abs(preds_orig - actuals).mean()
        l_acc = 100 * (1 - l_mae / actuals.mean())
        print(f"LSTM    - MAE: {l_mae:.2f}, Approximate Accuracy: {l_acc:.2f}%")
        
    except Exception as e:
        print(f"Could not complete evaluation: {e}")

    print("\nTraining Phase Complete! ML components are ready for the dashboard.")
