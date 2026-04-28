import pytest
import pandas as pd
import numpy as np
import tempfile
import shutil
import torch
from src.ml.lstm_model import CrowdLSTMModel, SimpleMinMaxScaler


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────

@pytest.fixture
def temp_model_dir():
    """
    Temporary directory for any model checkpoints or artifacts written during tests.
    Deleted after each test automatically.
    """
    tmp = tempfile.mkdtemp(prefix="lstm_test_")
    yield tmp
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture
def sample_df():
    """Minimal synthetic DataFrame — NOT written to any real CSV."""
    df = pd.DataFrame({
        "ds": pd.date_range(start="2024-01-01", periods=20, freq="5min"),
        "y":  np.random.randint(10, 100, 20),
    })
    df["hour"] = df["ds"].dt.hour
    df["day"]  = df["ds"].dt.dayofweek
    return df


@pytest.fixture
def trained_model(sample_df, temp_model_dir):
    """Returns a freshly trained LSTM model using only temp/in-memory data."""
    model = CrowdLSTMModel(sequence_length=5, epochs=1)
    model.train(sample_df)
    return model


# ──────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────

def test_scaler_functionality():
    scaler = SimpleMinMaxScaler()
    data = np.array([10, 20, 30, 40, 50]).reshape(-1, 1)

    transformed = scaler.fit_transform(data)
    assert transformed.min() == -1.0
    assert transformed.max() ==  1.0

    inversed = scaler.inverse_transform(transformed)
    np.testing.assert_array_almost_equal(data, inversed)


def test_lstm_model_training(sample_df, temp_model_dir):
    model = CrowdLSTMModel(sequence_length=5, epochs=1)
    success = model.train(sample_df)

    assert success is True
    assert model.is_trained is True


def test_lstm_prediction_output(trained_model, sample_df):
    predictions = trained_model.predict(sample_df, periods=5)

    assert predictions is not None
    assert len(predictions) == 5
    assert all(isinstance(p, (float, np.float32, np.float64)) for p in predictions)
