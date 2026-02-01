"""Unit tests for Model classes."""
import pytest
import pandas as pd
import numpy as np
from ml_service.machine_learning.model import RandomForestModel, XGBoostModel, ModelFactory


@pytest.fixture
def sample_classification_data():
    """Create sample classification data."""
    np.random.seed(42)
    X_train = pd.DataFrame(np.random.randn(100, 5), columns=[f'feature_{i}' for i in range(5)])
    y_train = pd.Series(np.random.randint(0, 2, 100), name='target')
    X_test = pd.DataFrame(np.random.randn(20, 5), columns=[f'feature_{i}' for i in range(5)])
    y_test = pd.Series(np.random.randint(0, 2, 20), name='target')
    return X_train, X_test, y_train, y_test


class TestRandomForestModel:
    """Test suite for RandomForestModel."""
    
    def test_build_classifier(self):
        """Test building random forest classifier."""
        config = {'model_type': 'classifier', 'n_estimators': 10, 'random_state': 42}
        model = RandomForestModel(config)
        model.build_model()
        
        assert model.model is not None
        assert not model.is_trained
    
    def test_train_classifier(self, sample_classification_data):
        """Test training random forest classifier."""
        X_train, X_test, y_train, y_test = sample_classification_data
        
        config = {'model_type': 'classifier', 'n_estimators': 10, 'random_state': 42}
        model = RandomForestModel(config)
        metrics = model.train(X_train, y_train, X_test, y_test)
        
        assert model.is_trained
        assert 'train_score' in metrics
        assert 'val_score' in metrics
    
    def test_predict_classifier(self, sample_classification_data):
        """Test making predictions with random forest classifier."""
        X_train, X_test, y_train, y_test = sample_classification_data
        
        config = {'model_type': 'classifier', 'n_estimators': 10, 'random_state': 42}
        model = RandomForestModel(config)
        model.train(X_train, y_train)
        
        predictions = model.predict(X_test)
        
        assert len(predictions) == len(X_test)
        assert predictions.dtype in [np.int32, np.int64]
    
    def test_save_and_load_model(self, sample_classification_data, tmp_path):
        """Test saving and loading model."""
        X_train, _, y_train, _ = sample_classification_data
        
        config = {'model_type': 'classifier', 'n_estimators': 10, 'random_state': 42}
        model = RandomForestModel(config)
        model.train(X_train, y_train)
        
        model_path = tmp_path / "test_model.joblib"
        model.save_model(str(model_path))
        
        new_model = RandomForestModel(config)
        new_model.load_model(str(model_path))
        
        assert new_model.is_trained


class TestModelFactory:
    """Test suite for ModelFactory."""
    
    def test_create_random_forest(self):
        """Test creating random forest model via factory."""
        model = ModelFactory.create_model('random_forest', {'model_type': 'classifier'})
        
        assert isinstance(model, RandomForestModel)
    
    def test_create_invalid_model(self):
        """Test creating invalid model type."""
        with pytest.raises(ValueError):
            ModelFactory.create_model('invalid_model', {})
