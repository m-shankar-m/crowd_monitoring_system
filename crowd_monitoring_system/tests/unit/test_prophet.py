import pytest
import pandas as pd
import numpy as np
from src.ml.prophet_model import ForecastModel

def test_prophet_training():
    model = ForecastModel()
    # Prophet requires 'ds' and 'y' columns
    df = pd.DataFrame({
        'ds': pd.date_range(start='2024-01-01', periods=10, freq='5min'),
        'y': np.random.randint(10, 100, 10)
    })
    
    success = model.train(df)
    assert success is True
    assert model.model is not None

def test_prophet_prediction_output():
    model = ForecastModel()
    df = pd.DataFrame({
        'ds': pd.date_range(start='2024-01-01', periods=10, freq='5min'),
        'y': np.random.randint(10, 100, 10)
    })
    model.train(df)
    
    forecast = model.predict(periods=5)
    assert forecast is not None
    assert len(forecast) == 5
    assert 'ds' in forecast.columns
    assert 'yhat' in forecast.columns
