"""Machine Learning model definitions and utilities."""
import logging
from typing import Optional, Dict, Any, Union
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
import joblib
from pathlib import Path


logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """Abstract base class for ML models."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.model = None
        self.is_trained = False
    
    @abstractmethod
    def build_model(self) -> None:
        """Build the model architecture."""
        pass
    
    @abstractmethod
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, 
             X_val: Optional[pd.DataFrame] = None, 
             y_val: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Train the model."""
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        pass
    
    def save_model(self, path: str) -> None:
        """Save model to disk."""
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self.model, path)
            logger.info(f"Model saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise
    
    def load_model(self, path: str) -> None:
        """Load model from disk."""
        try:
            self.model = joblib.load(path)
            self.is_trained = True
            logger.info(f"Model loaded from {path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise


class RandomForestModel(BaseModel):
    """Random Forest model implementation."""
    
    def build_model(self) -> None:
        """Build Random Forest model."""
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        
        model_type = self.config.get('model_type', 'classifier')
        n_estimators = self.config.get('n_estimators', 100)
        max_depth = self.config.get('max_depth', None)
        random_state = self.config.get('random_state', 42)
        
        if model_type == 'classifier':
            self.model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state
            )
        else:
            self.model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state
            )
        
        logger.info(f"Built Random Forest {model_type}")
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
             X_val: Optional[pd.DataFrame] = None,
             y_val: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Train Random Forest model."""
        try:
            if self.model is None:
                self.build_model()
            
            self.model.fit(X_train, y_train)
            self.is_trained = True
            
            train_score = self.model.score(X_train, y_train)
            metrics = {'train_score': train_score}
            
            if X_val is not None and y_val is not None:
                val_score = self.model.score(X_val, y_val)
                metrics['val_score'] = val_score
            
            logger.info(f"Model trained successfully. Metrics: {metrics}")
            return metrics
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using Random Forest."""
        try:
            if not self.is_trained:
                raise ValueError("Model is not trained yet")
            
            predictions = self.model.predict(X)
            logger.info(f"Generated {len(predictions)} predictions")
            return predictions
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise


class XGBoostModel(BaseModel):
    """XGBoost model implementation."""
    
    def build_model(self) -> None:
        """Build XGBoost model."""
        try:
            import xgboost as xgb
            
            model_type = self.config.get('model_type', 'classifier')
            n_estimators = self.config.get('n_estimators', 100)
            max_depth = self.config.get('max_depth', 6)
            learning_rate = self.config.get('learning_rate', 0.1)
            
            if model_type == 'classifier':
                self.model = xgb.XGBClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate
                )
            else:
                self.model = xgb.XGBRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate
                )
            
            logger.info(f"Built XGBoost {model_type}")
        except ImportError:
            logger.error("XGBoost not installed. Install with: pip install xgboost")
            raise
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
             X_val: Optional[pd.DataFrame] = None,
             y_val: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Train XGBoost model."""
        try:
            if self.model is None:
                self.build_model()
            
            eval_set = [(X_train, y_train)]
            if X_val is not None and y_val is not None:
                eval_set.append((X_val, y_val))
            
            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,
                verbose=False
            )
            self.is_trained = True
            
            metrics = {'train_score': self.model.score(X_train, y_train)}
            if X_val is not None:
                metrics['val_score'] = self.model.score(X_val, y_val)
            
            logger.info(f"Model trained successfully. Metrics: {metrics}")
            return metrics
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using XGBoost."""
        try:
            if not self.is_trained:
                raise ValueError("Model is not trained yet")
            
            predictions = self.model.predict(X)
            logger.info(f"Generated {len(predictions)} predictions")
            return predictions
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise


class ModelFactory:
    """Factory for creating ML models."""
    
    @staticmethod
    def create_model(model_type: str, config: Optional[Dict[str, Any]] = None) -> BaseModel:
        """Create and return appropriate model."""
        models = {
            'random_forest': RandomForestModel,
            'xgboost': XGBoostModel,
        }
        
        model_class = models.get(model_type.lower())
        if not model_class:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        return model_class(config)
