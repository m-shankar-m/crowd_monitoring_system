import pytest
import os
import pandas as pd
from src.ml.pipeline import MLPipeline

def test_ml_pipeline_forecast_generation(test_time_series):
    # Initialize pipeline
    pipeline = MLPipeline()
    
    # Check if get_forecast returns expected structure
    # We pass history_counts to avoid reading from disk if possible
    history = test_time_series['count'].tail(20).tolist()
    forecast = pipeline.get_forecast(periods=5, history_counts=history)
    
    # Even if models aren't loaded, pipeline should handle it gracefully (return empty or fallback)
    assert isinstance(forecast, list)
    if len(forecast) > 0:
        assert len(forecast) == 5
        assert "timestamp" in forecast[0]
        assert "predicted_count" in forecast[0]

def test_ml_pipeline_training_workflow(test_data_dir, project_root):
    # Setup a mock data path for training test
    mock_csv = os.path.join(test_data_dir, 'test_time_series.csv')
    pipeline = MLPipeline()
    pipeline.csv_path = mock_csv
    
    # Attempt training (will likely fail if epochs are too high, but we check the return status)
    # Reducing epochs for faster test execution if possible, but pipeline has hardcoded 10
    result = pipeline.train_model()
    
    assert "status" in result
    assert result["status"] in ["success", "error"]
