"""Pytest configuration and fixtures."""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path


@pytest.fixture(scope="session")
def test_data_dir(tmp_path_factory):
    """Create a temporary directory for test data."""
    return tmp_path_factory.mktemp("test_data")


@pytest.fixture
def sample_csv_data(test_data_dir):
    """Create a sample CSV file for testing."""
    df = pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100),
        'feature3': np.random.randint(0, 10, 100),
        'target': np.random.randint(0, 2, 100)
    })
    
    csv_path = test_data_dir / "sample_data.csv"
    df.to_csv(csv_path, index=False)
    
    return str(csv_path)


@pytest.fixture
def mock_model_path(tmp_path):
    """Create a mock model path."""
    return tmp_path / "models" / "test_model.joblib"


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
