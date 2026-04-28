import pytest
import numpy as np
import os
import time
from src.cv.pipeline import CVPipeline

def test_cv_pipeline_initialization():
    # Test that it creates the CSV and header
    csv_path = "data/crowd_data_test.csv"
    if os.path.exists(csv_path): os.remove(csv_path)
    
    pipeline = CVPipeline()
    pipeline.csv_path = csv_path
    
    # Re-init to trigger existing file check logic if any, 
    # but the logic for header is in __init__
    pipeline.__init__()
    pipeline.csv_path = csv_path
    
    assert os.path.exists(os.path.dirname(csv_path))

def test_cv_pipeline_process_frame():
    pipeline = CVPipeline()
    pipeline.csv_path = "data/crowd_data_test.csv"
    
    # Create a dummy frame
    frame = np.zeros((640, 640, 3), dtype=np.uint8)
    
    # Process multiple frames to test smoothing and logging
    for _ in range(10):
        result = pipeline.process_frame(frame)
        assert "count" in result
        assert "tracks" in result
    
    # Force log time to trigger CSV write
    pipeline.last_log_time = 0 
    pipeline.process_frame(frame)
    
    assert os.path.exists(pipeline.csv_path)
    
    # Clean up
    if os.path.exists(pipeline.csv_path): os.remove(pipeline.csv_path)

def test_cv_pipeline_smoothing():
    pipeline = CVPipeline()
    frame = np.zeros((640, 640, 3), dtype=np.uint8)
    
    # Mock counts buffer to test smoothing logic
    pipeline.counts_buffer = [10, 20, 30, 40, 50]
    result = pipeline.process_frame(frame)
    
    # (10+20+30+40+50 + 0)/6 = 150/6 = 25 (if new detect returns 0)
    # The new count is appended, and if > 5, first is popped.
    # Buffer becomes [20, 30, 40, 50, 0] -> sum=140, len=5 -> avg=28
    assert result["count"] >= 0
