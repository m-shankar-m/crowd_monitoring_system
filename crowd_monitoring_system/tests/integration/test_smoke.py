import os
import pytest
from dotenv import load_dotenv

def test_environment_variables():
    """Verify that essential environment variables are loadable."""
    load_dotenv()
    # Check for some common variables or at least that load_dotenv doesn't crash
    # We can check for specific ones if we know them, e.g., CAPACITY_THRESHOLD
    threshold = os.getenv("CAPACITY_THRESHOLD", "50")
    assert threshold is not None
    assert int(threshold) > 0

def test_critical_directories():
    """Verify that necessary directories exist for runtime."""
    directories = ["data", "models", "logs", "src"]
    for d in directories:
        assert os.path.exists(d), f"Critical directory {d} is missing"

def test_model_files_present():
    """Check if model files are in the expected location."""
    # This is a 'smoke' check for deployment readiness
    model_dir = "models"
    # Even if empty, the dir should exist. 
    # If we expect specific files like 'yolov8n.pt', we could check here.
    assert os.path.isdir(model_dir)

def test_logging_readiness():
    """Verify that the system can write to logs."""
    log_file = "logs/test_smoke.log"
    os.makedirs("logs", exist_ok=True)
    with open(log_file, "w") as f:
        f.write("Smoke test log entry")
    assert os.path.exists(log_file)
    os.remove(log_file)
