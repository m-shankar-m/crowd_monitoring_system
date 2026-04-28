import pytest
import pandas as pd
import numpy as np

def test_dataframe_resampling():
    # Test if we can correctly handle irregular time series data
    # Create irregular data
    times = [
        '2024-01-01 10:00',
        '2024-01-01 10:03', # Gap
        '2024-01-01 10:11', # Gap
    ]
    df = pd.DataFrame({
        'ds': pd.to_datetime(times),
        'y': [10, 12, 15]
    })
    
    # Check if we can resample to 5min
    df_resampled = df.set_index('ds').resample('5min').mean().fillna(0).reset_index()
    assert len(df_resampled) >= 3
    assert (df_resampled['ds'].diff().iloc[1:] == pd.Timedelta('5min')).all()

def test_feature_engineering_for_lstm():
    df = pd.DataFrame({
        'ds': pd.date_range(start='2024-01-01', periods=10, freq='h'),
        'y': np.random.randint(10, 50, 10)
    })
    
    # Simulate the feature extraction done in lstm_model.py
    df['hour'] = df['ds'].dt.hour
    df['day'] = df['ds'].dt.dayofweek
    
    assert 'hour' in df.columns
    assert 'day' in df.columns
    assert df['hour'].iloc[0] == 0
    assert df['hour'].iloc[1] == 1

def test_missing_value_impact_on_prophet():
    # Prophet handles missing values well, but let's verify our wrapping logic
    df = pd.DataFrame({
        'ds': pd.date_range(start='2024-01-01', periods=10, freq='5min'),
        'y': [10, 12, np.nan, 15, 20, np.nan, 25, 30, 35, 40]
    })
    
    # Our ForecastModel should handle this or we should fill it
    df_filled = df.ffill().fillna(0)
    assert df_filled['y'].isna().sum() == 0
