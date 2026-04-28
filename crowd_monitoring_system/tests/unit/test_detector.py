import pytest
import numpy as np
import os
from src.cv.detector import PersonDetector

def test_detector_initialization():
    detector = PersonDetector()
    assert detector.model is not None

def test_detector_output_format():
    detector = PersonDetector()
    # Create a blank black image
    image = np.zeros((640, 640, 3), dtype=np.uint8)
    boxes = detector.detect(image)
    
    assert isinstance(boxes, list)
    if len(boxes) > 0:
        for box in boxes:
            assert len(box) == 5  # x1, y1, x2, y2, conf
            assert all(isinstance(val, (int, float)) for val in box)
