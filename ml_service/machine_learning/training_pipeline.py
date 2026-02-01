"""Training pipeline orchestration."""
import logging
from typing import Dict, Any, Optional
import pandas as pd
from pathlib import Path
import json
from datetime import datetime

from ml_service.data_layer.data_connector import DataConnectorFactory
from ml_service.data_layer.object_connector import ObjectConnectorFactory
from ml_service.machine_learning.data_processor import DataProcessor
from ml_service.machine_learning.model import ModelFactory
from ml_service.machine_learning.cross_validator import CrossValidator


logger = logging.getLogger(__name__)


class TrainingPipeline:
    """Orchestrate the complete ML training pipeline."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_processor = DataProcessor(config.get('data_processing', {}))
        self.cross_validator = CrossValidator(
            n_splits=config.get('cv_splits', 5),
            random_state=config.get('random_state', 42)
        )
        self.model = None
        self.training_metrics = {}
    
    def load_data_from_source(self) -> pd.DataFrame:
        """Load data from configured source."""
        try:
            data_config = self.config.get('data_source', {})
            source_type = data_config.get('type', 'file')
            
            if source_type == 'database':
                connector = DataConnectorFactory.create_connector(
                    data_config.get('connector_type'),
                    data_config.get('connection_config')
                )
                connector.connect()
                df = connector.execute_query(data_config.get('query'))
                connector.disconnect()
            elif source_type == 'object_storage':
                connector = ObjectConnectorFactory.create_connector(
                    data_config.get('connector_type'),
                    data_config.get('connection_config')
                )
                connector.connect()
                local_path = data_config.get('local_path', 'temp_data.csv')
                connector.download_file(data_config.get('remote_path'), local_path)
                df = pd.read_csv(local_path)
            else:
                # Load from local file
                df = self.data_processor.load_data(data_config.get('path'))
            
            logger.info(f"Loaded data with shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise
    
    def preprocess_data(self, df: pd.DataFrame) -> tuple:
        """Preprocess the data."""
        try:
            target_column = self.config.get('target_column')
            
            X_train, X_test, y_train, y_test = self.data_processor.preprocess_pipeline(
                df, target_column, fit=True
            )
            
            logger.info("Data preprocessing completed")
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logger.error(f"Data preprocessing failed: {e}")
            raise
    
    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                   X_val: Optional[pd.DataFrame] = None,
                   y_val: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Train the model."""
        try:
            model_config = self.config.get('model', {})
            model_type = model_config.get('type', 'random_forest')
            
            self.model = ModelFactory.create_model(model_type, model_config)
            
            metrics = self.model.train(X_train, y_train, X_val, y_val)
            
            logger.info(f"Model training completed. Metrics: {metrics}")
            return metrics
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            raise
    
    def validate_model(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Validate model using cross-validation."""
        try:
            cv_config = self.config.get('cross_validation', {})
            cv_type = cv_config.get('type', 'kfold')
            scoring = cv_config.get('scoring', 'accuracy')
            
            if cv_type == 'kfold':
                results = self.cross_validator.k_fold_validation(
                    self.model.model, X, y, scoring
                )
            elif cv_type == 'stratified':
                results = self.cross_validator.stratified_k_fold_validation(
                    self.model.model, X, y, scoring
                )
            else:
                raise ValueError(f"Unsupported CV type: {cv_type}")
            
            logger.info("Model validation completed")
            return results
        except Exception as e:
            logger.error(f"Model validation failed: {e}")
            raise
    
    def evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Evaluate model on test set."""
        try:
            predictions = self.model.predict(X_test)
            
            task_type = self.config.get('task_type', 'classification')
            metrics = self.cross_validator.validate_predictions(
                y_test.values, predictions, task_type
            )
            
            logger.info(f"Model evaluation completed. Metrics: {metrics}")
            return metrics
        except Exception as e:
            logger.error(f"Model evaluation failed: {e}")
            raise
    
    def save_model(self, output_path: Optional[str] = None) -> str:
        """Save trained model."""
        try:
            if output_path is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_path = f"models/model_{timestamp}.joblib"
            
            self.model.save_model(output_path)
            
            # Save training metadata
            metadata_path = output_path.replace('.joblib', '_metadata.json')
            metadata = {
                'config': self.config,
                'metrics': self.training_metrics,
                'timestamp': datetime.now().isoformat(),
                'feature_columns': self.data_processor.feature_columns
            }
            
            Path(metadata_path).parent.mkdir(parents=True, exist_ok=True)
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Model and metadata saved to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise
    
    def run_pipeline(self) -> Dict[str, Any]:
        """Execute the complete training pipeline."""
        try:
            logger.info("Starting training pipeline...")
            
            # Load data
            df = self.load_data_from_source()
            
            # Preprocess data
            X_train, X_test, y_train, y_test = self.preprocess_data(df)
            
            # Train model
            train_metrics = self.train_model(X_train, y_train)
            self.training_metrics['training'] = train_metrics
            
            # Validate model
            if self.config.get('run_cross_validation', True):
                cv_metrics = self.validate_model(X_train, y_train)
                self.training_metrics['cross_validation'] = cv_metrics
            
            # Evaluate on test set
            test_metrics = self.evaluate_model(X_test, y_test)
            self.training_metrics['test'] = test_metrics
            
            # Save model
            model_path = self.save_model(self.config.get('output_path'))
            self.training_metrics['model_path'] = model_path
            
            logger.info("Training pipeline completed successfully")
            return self.training_metrics
        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            raise
