"""Unit tests for CrossValidator."""
import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from ml_service.machine_learning.cross_validator import CrossValidator


@pytest.fixture
def sample_classification_data():
    """Create sample classification data."""
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(100, 5), columns=[f'feature_{i}' for i in range(5)])
    y = pd.Series(np.random.randint(0, 2, 100), name='target')
    return X, y


@pytest.fixture
def cross_validator():
    """Create a CrossValidator instance."""
    return CrossValidator(n_splits=3, random_state=42)


class TestCrossValidator:
    """Test suite for CrossValidator class."""
    
    def test_k_fold_validation(self, cross_validator, sample_classification_data):
        """Test K-Fold cross-validation."""
        X, y = sample_classification_data
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        
        results = cross_validator.k_fold_validation(model, X, y, scoring='accuracy')
        
        assert 'scores' in results
        assert 'mean_score' in results
        assert 'std_score' in results
        assert len(results['scores']) == 3
    
    def test_stratified_k_fold_validation(self, cross_validator, sample_classification_data):
        """Test Stratified K-Fold cross-validation."""
        X, y = sample_classification_data
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        
        results = cross_validator.stratified_k_fold_validation(model, X, y, scoring='accuracy')
        
        assert 'scores' in results
        assert 'mean_score' in results
        assert len(results['scores']) == 3
    
    def test_evaluate_classification(self, cross_validator):
        """Test classification evaluation metrics."""
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 0, 1])
        
        metrics = cross_validator.evaluate_classification(y_true, y_pred)
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        assert 0 <= metrics['accuracy'] <= 1
    
    def test_evaluate_regression(self, cross_validator):
        """Test regression evaluation metrics."""
        y_true = np.array([1.5, 2.0, 3.5, 4.0, 5.5])
        y_pred = np.array([1.6, 2.1, 3.4, 4.2, 5.3])
        
        metrics = cross_validator.evaluate_regression(y_true, y_pred)
        
        assert 'mse' in metrics
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'r2_score' in metrics
        assert metrics['mse'] >= 0
    
    def test_compare_models(self, cross_validator, sample_classification_data):
        """Test comparing multiple models."""
        X, y = sample_classification_data
        
        models = {
            'rf_10': RandomForestClassifier(n_estimators=10, random_state=42),
            'rf_20': RandomForestClassifier(n_estimators=20, random_state=42),
        }
        
        comparison = cross_validator.compare_models(models, X, y, cv_type='kfold')
        
        assert isinstance(comparison, pd.DataFrame)
        assert len(comparison) == 2
        assert 'model' in comparison.columns
        assert 'mean_score' in comparison.columns
