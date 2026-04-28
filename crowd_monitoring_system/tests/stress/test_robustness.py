import pytest
import pandas as pd
import numpy as np
from src.ml.lstm_model import CrowdLSTMModel

def test_lstm_robustness_to_spikes():
    model = CrowdLSTMModel(sequence_length=5, epochs=1)
    
    # Train on normal data
    df = pd.DataFrame({
        'ds': pd.date_range(start='2024-01-01', periods=50, freq='5min'),
        'y': np.random.randint(10, 20, 50)
    })
    df['hour'] = df['ds'].dt.hour
    df['day'] = df['ds'].dt.dayofweek
    model.train(df)
    
    # Predict with a massive spike in input
    spike_df = pd.DataFrame({
        'ds': pd.date_range(start='2024-01-01 04:15:00', periods=5, freq='5min'),
        'y': [15, 16, 1000, 14, 15]
    })
    spike_df['hour'] = spike_df['ds'].dt.hour
    spike_df['day'] = spike_df['ds'].dt.dayofweek
    prediction = model.predict(spike_df, periods=1)
    
    # Ensure it doesn't crash and returns a finite value
    assert prediction is not None
    assert np.isfinite(prediction[0])

def test_lstm_robustness_to_zeros():
    model = CrowdLSTMModel(sequence_length=5, epochs=1)
    
    # Train on normal data
    df = pd.DataFrame({
        'ds': pd.date_range(start='2024-01-01', periods=50, freq='5min'),
        'y': np.random.randint(10, 20, 50)
    })
    df['hour'] = df['ds'].dt.hour
    df['day'] = df['ds'].dt.dayofweek
    model.train(df)
    
    # Input with all zeros
    zero_df = pd.DataFrame({
        'ds': pd.date_range(start='2024-01-01 04:15:00', periods=5, freq='5min'),
        'y': [0, 0, 0, 0, 0]
    })
    zero_df['hour'] = zero_df['ds'].dt.hour
    zero_df['day'] = zero_df['ds'].dt.dayofweek
    prediction = model.predict(zero_df, periods=1)
    
    assert prediction is not None
    assert prediction[0] >= 0  # Should not predict negative counts
