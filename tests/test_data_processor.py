"""Unit tests for DataProcessor."""
import pytest
import pandas as pd
import numpy as np
from ml_service.machine_learning.data_processor import DataProcessor


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        'feature1': [1, 2, np.nan, 4, 5],
        'feature2': [10, 20, 30, np.nan, 50],
        'category': ['A', 'B', 'A', 'C', 'B'],
        'target': [0, 1, 0, 1, 0]
    })


@pytest.fixture
def data_processor():
    """Create a DataProcessor instance."""
    return DataProcessor()


class TestDataProcessor:
    """Test suite for DataProcessor class."""
    
    def test_load_data_from_csv(self, tmp_path, sample_dataframe):
        """Test loading data from CSV file."""
        csv_file = tmp_path / "test_data.csv"
        sample_dataframe.to_csv(csv_file, index=False)
        
        processor = DataProcessor()
        df = processor.load_data(str(csv_file))
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5
        assert list(df.columns) == ['feature1', 'feature2', 'category', 'target']
    
    def test_load_data_from_dataframe(self, data_processor, sample_dataframe):
        """Test loading data from DataFrame."""
        df = data_processor.load_data(sample_dataframe)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5
    
    def test_handle_missing_values_mean(self, data_processor, sample_dataframe):
        """Test handling missing values with mean strategy."""
        df = data_processor.handle_missing_values(sample_dataframe, strategy='mean')
        
        assert df['feature1'].isna().sum() == 0
        assert df['feature2'].isna().sum() == 0
        assert df['feature1'].iloc[2] == pytest.approx(3.0, rel=1e-5)
    
    def test_handle_missing_values_drop(self, data_processor, sample_dataframe):
        """Test handling missing values by dropping."""
        df = data_processor.handle_missing_values(sample_dataframe, strategy='drop')
        
        assert len(df) == 3
        assert df.isna().sum().sum() == 0
    
    def test_encode_categorical_features(self, data_processor, sample_dataframe):
        """Test encoding categorical features."""
        df = data_processor.encode_categorical_features(sample_dataframe)
        
        assert 'category' in df.columns
        assert df['category'].dtype in [np.int32, np.int64]
    
    def test_scale_features(self, data_processor, sample_dataframe):
        """Test scaling numerical features."""
        df_clean = data_processor.handle_missing_values(sample_dataframe, strategy='mean')
        df_scaled = data_processor.scale_features(df_clean, method='standard')
        
        # Check that features are scaled
        assert 'feature1' in df_scaled.columns
        assert 'feature2' in df_scaled.columns
    
    def test_split_data(self, data_processor, sample_dataframe):
        """Test splitting data into train and test sets."""
        df_clean = data_processor.handle_missing_values(sample_dataframe, strategy='drop')
        X_train, X_test, y_train, y_test = data_processor.split_data(
            df_clean, target_column='target', test_size=0.33
        )
        
        assert len(X_train) + len(X_test) == len(df_clean)
        assert len(y_train) == len(X_train)
        assert len(y_test) == len(X_test)
        assert 'target' not in X_train.columns
        assert 'target' not in X_test.columns
