import pytest
import pandas as pd
import numpy as np
import tempfile
import shutil
from src.ml.prophet_model import ForecastModel


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────

@pytest.fixture
def temp_model_dir():
    """
    Temporary directory for any Prophet serialization artifacts.
    Cleaned up after every test.
    """
    tmp = tempfile.mkdtemp(prefix="prophet_test_")
    yield tmp
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture
def sample_df():
    """Synthetic DataFrame — lives only in memory, never touches the real CSV."""
    return pd.DataFrame({
        "ds": pd.date_range(start="2024-01-01", periods=10, freq="5min"),
        "y":  np.random.randint(10, 100, 10),
    })


@pytest.fixture
def trained_model(sample_df):
    """Returns a trained ForecastModel using only in-memory data."""
    model = ForecastModel()
    model.train(sample_df)
    return model


# ──────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────

def test_prophet_training(sample_df):
    model = ForecastModel()
    success = model.train(sample_df)

    assert success is True
    assert model.model is not None


def test_prophet_prediction_output(trained_model):
    forecast = trained_model.predict(periods=5)

    assert forecast is not None
    assert len(forecast) == 5
    assert "ds"   in forecast.columns
    assert "yhat" in forecast.columns
