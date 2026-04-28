import pytest
import numpy as np
from src.cv.detector import PersonDetector


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────

@pytest.fixture(scope="module")
def detector():
    """
    Shared detector instance for the module.
    Detector tests don't write to any dataset, so no temp dir is needed.
    """
    return PersonDetector()


# ──────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────

def test_detector_initialization(detector):
    assert detector.model is not None


def test_detector_output_format(detector):
    image = np.zeros((640, 640, 3), dtype=np.uint8)
    boxes = detector.detect(image)

    assert isinstance(boxes, list)
    for box in boxes:
        assert len(box) == 5          # x1, y1, x2, y2, conf
        assert all(isinstance(v, (int, float)) for v in box)
