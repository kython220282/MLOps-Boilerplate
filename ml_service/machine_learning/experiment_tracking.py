"""MLflow experiment tracking integration."""
import logging
from typing import Dict, Any, Optional
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from pathlib import Path

logger = logging.getLogger(__name__)


class MLflowTracker:
    """MLflow experiment tracking and model registry."""
    
    def __init__(self, tracking_uri: Optional[str] = None,
                 experiment_name: str = "ml_service_experiments"):
        """
        Initialize MLflow tracker.
        
        Args:
            tracking_uri: MLflow tracking server URI
            experiment_name: Name of the experiment
        """
        self.tracking_uri = tracking_uri or "http://localhost:5000"
        self.experiment_name = experiment_name
        
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        
        logger.info(f"MLflow tracking URI: {self.tracking_uri}")
        logger.info(f"Experiment: {self.experiment_name}")
    
    def start_run(self, run_name: Optional[str] = None) -> mlflow.ActiveRun:
        """
        Start a new MLflow run.
        
        Args:
            run_name: Name for the run
            
        Returns:
            Active MLflow run
        """
        return mlflow.start_run(run_name=run_name)
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """
        Log parameters to MLflow.
        
        Args:
            params: Dictionary of parameters
        """
        mlflow.log_params(params)
        logger.info(f"Logged {len(params)} parameters")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """
        Log metrics to MLflow.
        
        Args:
            metrics: Dictionary of metrics
            step: Step number for the metrics
        """
        mlflow.log_metrics(metrics, step=step)
        logger.info(f"Logged {len(metrics)} metrics")
    
    def log_artifact(self, artifact_path: str) -> None:
        """
        Log an artifact to MLflow.
        
        Args:
            artifact_path: Path to the artifact
        """
        mlflow.log_artifact(artifact_path)
        logger.info(f"Logged artifact: {artifact_path}")
    
    def log_model(self, model: Any, artifact_path: str = "model",
                  registered_model_name: Optional[str] = None) -> None:
        """
        Log a model to MLflow.
        
        Args:
            model: Trained model
            artifact_path: Path within run for the model
            registered_model_name: Name for model registry
        """
        # Detect model type and log accordingly
        model_type = type(model).__name__
        
        if 'RandomForest' in model_type or 'sklearn' in str(type(model).__module__):
            mlflow.sklearn.log_model(
                model, artifact_path,
                registered_model_name=registered_model_name
            )
        elif 'XGB' in model_type:
            mlflow.xgboost.log_model(
                model, artifact_path,
                registered_model_name=registered_model_name
            )
        else:
            # Generic pickle logging
            mlflow.sklearn.log_model(
                model, artifact_path,
                registered_model_name=registered_model_name
            )
        
        logger.info(f"Logged model to {artifact_path}")
    
    def log_training_run(self, config: Dict[str, Any], metrics: Dict[str, float],
                        model: Any, model_name: Optional[str] = None) -> str:
        """
        Log complete training run to MLflow.
        
        Args:
            config: Training configuration
            metrics: Training metrics
            model: Trained model
            model_name: Name for model registry
            
        Returns:
            Run ID
        """
        with mlflow.start_run() as run:
            # Log parameters
            self.log_params(config)
            
            # Log metrics
            self.log_metrics(metrics)
            
            # Log model
            self.log_model(model, registered_model_name=model_name)
            
            # Log tags
            mlflow.set_tag("model_type", config.get('model', {}).get('type', 'unknown'))
            mlflow.set_tag("task_type", config.get('task_type', 'unknown'))
            
            logger.info(f"Completed MLflow run: {run.info.run_id}")
            return run.info.run_id
    
    def load_model(self, model_uri: str) -> Any:
        """
        Load a model from MLflow.
        
        Args:
            model_uri: URI of the model (e.g., 'runs:/<run_id>/model')
            
        Returns:
            Loaded model
        """
        model = mlflow.sklearn.load_model(model_uri)
        logger.info(f"Loaded model from {model_uri}")
        return model
    
    def register_model(self, model_uri: str, model_name: str) -> None:
        """
        Register a model in MLflow Model Registry.
        
        Args:
            model_uri: URI of the model
            model_name: Name for the registered model
        """
        mlflow.register_model(model_uri, model_name)
        logger.info(f"Registered model {model_name}")
    
    def transition_model_stage(self, model_name: str, version: int,
                              stage: str = "Production") -> None:
        """
        Transition model to a different stage.
        
        Args:
            model_name: Name of the registered model
            version: Model version
            stage: Target stage ('Staging', 'Production', 'Archived')
        """
        from mlflow.tracking import MlflowClient
        
        client = MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage
        )
        logger.info(f"Transitioned {model_name} v{version} to {stage}")


def enable_autolog(framework: str = 'sklearn') -> None:
    """
    Enable MLflow autologging for a framework.
    
    Args:
        framework: Framework name ('sklearn', 'xgboost', 'tensorflow', etc.)
    """
    if framework == 'sklearn':
        mlflow.sklearn.autolog()
    elif framework == 'xgboost':
        mlflow.xgboost.autolog()
    elif framework == 'tensorflow':
        mlflow.tensorflow.autolog()
    elif framework == 'pytorch':
        mlflow.pytorch.autolog()
    
    logger.info(f"Enabled autologging for {framework}")
