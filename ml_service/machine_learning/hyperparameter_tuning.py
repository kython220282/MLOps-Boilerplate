"""Hyperparameter tuning utilities."""
import logging
from typing import Dict, Any, Optional, List, Union, Callable
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import optuna
from optuna.samplers import TPESampler

logger = logging.getLogger(__name__)


class HyperparameterTuner:
    """Hyperparameter tuning using GridSearch, RandomSearch, or Optuna."""
    
    def __init__(self, method: str = 'optuna', n_trials: int = 100, cv: int = 5):
        """
        Initialize hyperparameter tuner.
        
        Args:
            method: Tuning method ('grid', 'random', 'optuna')
            n_trials: Number of trials for Optuna
            cv: Number of cross-validation folds
        """
        self.method = method.lower()
        self.n_trials = n_trials
        self.cv = cv
        self.best_params = None
        self.best_score = None
        self.study = None
    
    def grid_search(self, model: Any, X: pd.DataFrame, y: pd.Series,
                   param_grid: Dict[str, List], scoring: str = 'accuracy') -> Dict[str, Any]:
        """
        Perform grid search for hyperparameter tuning.
        
        Args:
            model: Model to tune
            X: Training features
            y: Training target
            param_grid: Parameter grid
            scoring: Scoring metric
            
        Returns:
            Dictionary with best parameters and score
        """
        try:
            grid_search = GridSearchCV(
                model, param_grid, cv=self.cv, scoring=scoring,
                n_jobs=-1, verbose=1
            )
            
            grid_search.fit(X, y)
            
            self.best_params = grid_search.best_params_
            self.best_score = grid_search.best_score_
            
            logger.info(f"Best parameters: {self.best_params}")
            logger.info(f"Best score: {self.best_score:.4f}")
            
            return {
                'best_params': self.best_params,
                'best_score': self.best_score,
                'best_model': grid_search.best_estimator_,
                'cv_results': grid_search.cv_results_
            }
        except Exception as e:
            logger.error(f"Grid search failed: {e}")
            raise
    
    def random_search(self, model: Any, X: pd.DataFrame, y: pd.Series,
                     param_distributions: Dict[str, Any], scoring: str = 'accuracy',
                     n_iter: int = 100) -> Dict[str, Any]:
        """
        Perform random search for hyperparameter tuning.
        
        Args:
            model: Model to tune
            X: Training features
            y: Training target
            param_distributions: Parameter distributions
            scoring: Scoring metric
            n_iter: Number of iterations
            
        Returns:
            Dictionary with best parameters and score
        """
        try:
            random_search = RandomizedSearchCV(
                model, param_distributions, cv=self.cv, scoring=scoring,
                n_iter=n_iter, n_jobs=-1, verbose=1, random_state=42
            )
            
            random_search.fit(X, y)
            
            self.best_params = random_search.best_params_
            self.best_score = random_search.best_score_
            
            logger.info(f"Best parameters: {self.best_params}")
            logger.info(f"Best score: {self.best_score:.4f}")
            
            return {
                'best_params': self.best_params,
                'best_score': self.best_score,
                'best_model': random_search.best_estimator_,
                'cv_results': random_search.cv_results_
            }
        except Exception as e:
            logger.error(f"Random search failed: {e}")
            raise
    
    def optuna_search(self, model_class: Any, X: pd.DataFrame, y: pd.Series,
                     param_space: Union[Callable, Dict[str, Any]], scoring: str = 'accuracy',
                     direction: str = 'maximize') -> Dict[str, Any]:
        """
        Perform Optuna-based hyperparameter optimization.
        
        Args:
            model_class: Model class to instantiate
            X: Training features
            y: Training target
            param_space: Function that defines parameter search space
            scoring: Scoring metric
            direction: Optimization direction ('maximize' or 'minimize')
            
        Returns:
            Dictionary with best parameters and score
        """
        try:
            from sklearn.model_selection import cross_val_score
            
            def objective(trial):
                # Get parameters from search space
                # param_space can be either a callable or a dict
                if callable(param_space):
                    params = param_space(trial)
                else:
                    params = param_space
                
                # Create model with trial parameters
                model = model_class(**params)
                
                # Cross-validation score
                scores = cross_val_score(model, X, y, cv=self.cv, scoring=scoring, n_jobs=-1)
                
                return scores.mean()
            
            # Create study
            self.study = optuna.create_study(
                direction=direction,
                sampler=TPESampler(seed=42)
            )
            
            # Optimize
            self.study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)
            
            self.best_params = self.study.best_params
            self.best_score = self.study.best_value
            
            logger.info(f"Best parameters: {self.best_params}")
            logger.info(f"Best score: {self.best_score:.4f}")
            
            # Train final model with best parameters
            best_model = model_class(**self.best_params)
            best_model.fit(X, y)
            
            return {
                'best_params': self.best_params,
                'best_score': self.best_score,
                'best_model': best_model,
                'study': self.study
            }
        except Exception as e:
            logger.error(f"Optuna search failed: {e}")
            raise
    
    def tune(self, model: Any, X: pd.DataFrame, y: pd.Series,
            param_config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Tune hyperparameters using configured method.
        
        Args:
            model: Model to tune
            X: Training features
            y: Training target
            param_config: Parameter configuration
            **kwargs: Additional arguments for specific methods
            
        Returns:
            Dictionary with tuning results
        """
        if self.method == 'grid':
            return self.grid_search(model, X, y, param_config, **kwargs)
        elif self.method == 'random':
            return self.random_search(model, X, y, param_config, **kwargs)
        elif self.method == 'optuna':
            return self.optuna_search(model, X, y, param_config, **kwargs)
        else:
            raise ValueError(f"Unsupported tuning method: {self.method}")


def create_optuna_param_space_rf(trial):
    """
    Create Optuna parameter space for Random Forest.
    
    Args:
        trial: Optuna trial object
        
    Returns:
        Dictionary of parameters
    """
    return {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'random_state': 42
    }


def create_optuna_param_space_xgboost(trial):
    """
    Create Optuna parameter space for XGBoost.
    
    Args:
        trial: Optuna trial object
        
    Returns:
        Dictionary of parameters
    """
    return {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0, 1),
        'random_state': 42
    }
