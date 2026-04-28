import pytest
import os
import pandas as pd

@pytest.fixture
def project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@pytest.fixture
def test_data_dir():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_data')

@pytest.fixture
def test_time_series(test_data_dir):
    csv_path = os.path.join(test_data_dir, 'test_time_series.csv')
    return pd.read_csv(csv_path)
