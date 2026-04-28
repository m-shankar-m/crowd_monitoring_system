import pytest
import pandas as pd
import numpy as np
from src.ml.prophet_model import ForecastModel

def test_prophet_seasonality_handling():
    # Prophet is good with seasonality, let's ensure it doesn't crash on long forecasts
    model = ForecastModel()
    df = pd.DataFrame({
        'ds': pd.date_range(start='2024-01-01', periods=100, freq='h'),
        'y': np.random.randint(10, 50, 100)
    })
    
    model.train(df)
    forecast = model.predict(periods=24) # Predict 1 day ahead
    assert len(forecast) == 24
    assert forecast['yhat'].iloc[0] > 0
