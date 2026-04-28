import pytest
import pandas as pd
import numpy as np
from src.ml.lstm_model import CrowdLSTMModel

def test_lstm_full_cycle():
    # Test training, saving, loading, and predicting
    model = CrowdLSTMModel(sequence_length=5, epochs=1)
    df = pd.DataFrame({
        'ds': pd.date_range(start='2024-01-01', periods=30, freq='5min'),
        'y': np.random.randint(10, 100, 30)
    })
    df['hour'] = df['ds'].dt.hour
    df['day'] = df['ds'].dt.dayofweek
    
    # Train
    model.train(df)
    
    # Predict
    preds = model.predict(df, periods=5)
    assert len(preds) == 5
    
    # Save/Load (using temporary names)
    import os
    weights_path = "tests/test_data/temp_lstm.pth"
    scaler_path = "tests/test_data/temp_scaler.pkl"
    os.makedirs("tests/test_data", exist_ok=True)
    
    model.save(paths=(weights_path, scaler_path))
    
    new_model = CrowdLSTMModel(sequence_length=5)
    new_model.load(paths=(weights_path, scaler_path))
    
    assert new_model.is_trained is True
    
    # Clean up
    if os.path.exists(weights_path): os.remove(weights_path)
    if os.path.exists(scaler_path): os.remove(scaler_path)
