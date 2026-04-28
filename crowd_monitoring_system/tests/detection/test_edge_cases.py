import pytest
import numpy as np
from src.cv.detector import PersonDetector

@pytest.fixture
def detector():
    return PersonDetector()

def test_detect_black_image(detector):
    image = np.zeros((640, 640, 3), dtype=np.uint8)
    results = detector.detect(image)
    assert isinstance(results, list)
    # On a black image, it should ideally find 0 people
    assert len(results) == 0

def test_detect_white_image(detector):
    image = np.ones((640, 640, 3), dtype=np.uint8) * 255
    results = detector.detect(image)
    assert isinstance(results, list)

def test_detect_random_noise(detector):
    image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    results = detector.detect(image)
    assert isinstance(results, list)

def test_detect_unusual_size(detector):
    # Testing a very small image
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    results = detector.detect(image)
    assert isinstance(results, list)
    
    # Testing a non-square image
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    results = detector.detect(image)
    assert isinstance(results, list)
