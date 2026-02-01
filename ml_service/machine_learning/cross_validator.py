"""Cross-validation utilities for model evaluation."""
import logging
from typing import Dict, Any, Optional, List, Literal
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score
)


logger = logging.getLogger(__name__)


class CrossValidator:
    """Handle cross-validation and model evaluation."""
    
    def __init__(self, n_splits: int = 5, shuffle: bool = True, random_state: int = 42):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.results = {}
    
    def k_fold_validation(self, model: Any, X: pd.DataFrame, y: pd.Series,
                         scoring: str = 'accuracy') -> Dict[str, Any]:
        """Perform K-Fold cross-validation."""
        try:
            kfold = KFold(
                n_splits=self.n_splits,
                shuffle=self.shuffle,
                random_state=self.random_state
            )
            
            scores = cross_val_score(model, X, y, cv=kfold, scoring=scoring)
            
            results = {
                'scores': scores.tolist(),
                'mean_score': float(scores.mean()),
                'std_score': float(scores.std()),
                'scoring_metric': scoring
            }
            
            logger.info(f"K-Fold CV Results - Mean: {results['mean_score']:.4f} (+/- {results['std_score']:.4f})")
            return results
        except Exception as e:
            logger.error(f"K-Fold validation failed: {e}")
            raise
    
    def stratified_k_fold_validation(self, model: Any, X: pd.DataFrame, y: pd.Series,
                                    scoring: str = 'accuracy') -> Dict[str, Any]:
        """Perform Stratified K-Fold cross-validation."""
        try:
            skfold = StratifiedKFold(
                n_splits=self.n_splits,
                shuffle=self.shuffle,
                random_state=self.random_state
            )
            
            scores = cross_val_score(model, X, y, cv=skfold, scoring=scoring)
            
            results = {
                'scores': scores.tolist(),
                'mean_score': float(scores.mean()),
                'std_score': float(scores.std()),
                'scoring_metric': scoring
            }
            
            logger.info(f"Stratified K-Fold CV Results - Mean: {results['mean_score']:.4f} (+/- {results['std_score']:.4f})")
            return results
        except Exception as e:
            logger.error(f"Stratified K-Fold validation failed: {e}")
            raise
    
    def evaluate_classification(self, y_true: np.ndarray, y_pred: np.ndarray,
                               average: Literal['micro', 'macro', 'samples', 'weighted', 'binary'] = 'weighted') -> Dict[str, float]:
        """Evaluate classification model performance."""
        try:
            metrics = {
                'accuracy': float(accuracy_score(y_true, y_pred)),
                'precision': float(precision_score(y_true, y_pred, average=average, zero_division=0)),
                'recall': float(recall_score(y_true, y_pred, average=average, zero_division=0)),
                'f1_score': float(f1_score(y_true, y_pred, average=average, zero_division=0))
            }
            
            logger.info(f"Classification Metrics: {metrics}")
            return metrics
        except Exception as e:
            logger.error(f"Classification evaluation failed: {e}")
            raise
    
    def evaluate_regression(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Evaluate regression model performance."""
        try:
            metrics = {
                'mse': float(mean_squared_error(y_true, y_pred)),
                'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
                'mae': float(mean_absolute_error(y_true, y_pred)),
                'r2_score': float(r2_score(y_true, y_pred))
            }
            
            logger.info(f"Regression Metrics: {metrics}")
            return metrics
        except Exception as e:
            logger.error(f"Regression evaluation failed: {e}")
            raise
    
    def compare_models(self, models: Dict[str, Any], X: pd.DataFrame, y: pd.Series,
                      cv_type: str = 'kfold', scoring: str = 'accuracy') -> pd.DataFrame:
        """Compare multiple models using cross-validation."""
        try:
            results = []
            
            for model_name, model in models.items():
                logger.info(f"Evaluating {model_name}...")
                
                if cv_type == 'kfold':
                    cv_results = self.k_fold_validation(model, X, y, scoring)
                elif cv_type == 'stratified':
                    cv_results = self.stratified_k_fold_validation(model, X, y, scoring)
                else:
                    raise ValueError(f"Unsupported CV type: {cv_type}")
                
                results.append({
                    'model': model_name,
                    'mean_score': cv_results['mean_score'],
                    'std_score': cv_results['std_score']
                })
            
            comparison_df = pd.DataFrame(results).sort_values('mean_score', ascending=False)
            logger.info(f"Model comparison completed for {len(models)} models")
            
            return comparison_df
        except Exception as e:
            logger.error(f"Model comparison failed: {e}")
            raise
    
    def validate_predictions(self, y_true: np.ndarray, y_pred: np.ndarray,
                           task_type: str = 'classification') -> Dict[str, Any]:
        """Validate predictions based on task type."""
        try:
            if task_type == 'classification':
                metrics = self.evaluate_classification(y_true, y_pred)
            elif task_type == 'regression':
                metrics = self.evaluate_regression(y_true, y_pred)
            else:
                raise ValueError(f"Unsupported task type: {task_type}")
            
            return metrics
        except Exception as e:
            logger.error(f"Prediction validation failed: {e}")
            raise
