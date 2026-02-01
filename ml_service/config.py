"""Configuration management module."""
import os
import json
import yaml
from typing import Dict, Any, Optional
from pathlib import Path
from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class DatabaseConfig(BaseModel):
    """Database configuration."""
    type: str = Field(default="postgresql", description="Database type")
    host: str = Field(default="localhost")
    port: int = Field(default=5432)
    database: str = Field(default="ml_database")
    username: str = Field(default="")
    password: str = Field(default="")
    connection_pool_size: int = Field(default=10)


class ObjectStorageConfig(BaseModel):
    """Object storage configuration."""
    type: str = Field(default="s3", description="Storage type: s3, azure, gcs")
    bucket: str = Field(default="")
    access_key: Optional[str] = None
    secret_key: Optional[str] = None
    region: str = Field(default="us-east-1")
    connection_string: Optional[str] = None


class ModelConfig(BaseModel):
    """Model configuration."""
    type: str = Field(default="random_forest")
    model_type: str = Field(default="classifier")
    n_estimators: int = Field(default=100, ge=1)
    max_depth: Optional[int] = Field(default=None, ge=1)
    learning_rate: float = Field(default=0.1, gt=0, le=1)
    random_state: int = Field(default=42)


class DataProcessingConfig(BaseModel):
    """Data processing configuration."""
    missing_value_strategy: str = Field(default="mean")
    scaling_method: str = Field(default="standard")
    test_size: float = Field(default=0.2, gt=0, lt=1)
    random_state: int = Field(default=42)


class CrossValidationConfig(BaseModel):
    """Cross-validation configuration."""
    type: str = Field(default="kfold")
    n_splits: int = Field(default=5, ge=2)
    scoring: str = Field(default="accuracy")


class MLflowConfig(BaseModel):
    """MLflow configuration."""
    tracking_uri: str = Field(default="http://localhost:5000")
    experiment_name: str = Field(default="ml_service_experiments")
    artifact_location: Optional[str] = None
    enable_autolog: bool = Field(default=True)


class TrainingConfig(BaseModel):
    """Training pipeline configuration."""
    data_source: Dict[str, Any] = Field(default_factory=dict)
    target_column: str = Field(default="target")
    model: ModelConfig = Field(default_factory=ModelConfig)
    data_processing: DataProcessingConfig = Field(default_factory=DataProcessingConfig)
    cross_validation: CrossValidationConfig = Field(default_factory=CrossValidationConfig)
    task_type: str = Field(default="classification")
    output_path: Optional[str] = None
    run_cross_validation: bool = Field(default=True)
    random_state: int = Field(default=42)


class APIConfig(BaseModel):
    """API server configuration."""
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000, ge=1, le=65535)
    workers: int = Field(default=4, ge=1)
    reload: bool = Field(default=False)
    log_level: str = Field(default="info")


class MonitoringConfig(BaseModel):
    """Monitoring configuration."""
    enable_prometheus: bool = Field(default=True)
    prometheus_port: int = Field(default=9090)
    enable_data_drift: bool = Field(default=True)
    drift_threshold: float = Field(default=0.1)


class AppSettings(BaseSettings):
    """Application settings loaded from environment variables."""
    app_name: str = Field(default="ml_service", env="APP_NAME")
    environment: str = Field(default="development", env="ENVIRONMENT")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    debug: bool = Field(default=False, env="DEBUG")
    
    # Database
    db_type: str = Field(default="postgresql", env="DB_TYPE")
    db_host: str = Field(default="localhost", env="DB_HOST")
    db_port: int = Field(default=5432, env="DB_PORT")
    db_name: str = Field(default="ml_database", env="DB_NAME")
    db_user: str = Field(default="", env="DB_USER")
    db_password: str = Field(default="", env="DB_PASSWORD")
    
    # MLflow
    mlflow_tracking_uri: str = Field(default="http://localhost:5000", env="MLFLOW_TRACKING_URI")
    mlflow_experiment_name: str = Field(default="ml_service_experiments", env="MLFLOW_EXPERIMENT_NAME")
    
    class Config:
        env_file = ".env"
        case_sensitive = False


class ConfigLoader:
    """Load and validate configuration from files."""
    
    @staticmethod
    def load_from_json(config_path: str) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        with open(config_path, 'r') as f:
            return json.load(f)
    
    @staticmethod
    def load_from_yaml(config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    @staticmethod
    def load_training_config(config_path: str) -> TrainingConfig:
        """Load and validate training configuration."""
        if config_path.endswith('.json'):
            config_dict = ConfigLoader.load_from_json(config_path)
        elif config_path.endswith(('.yaml', '.yml')):
            config_dict = ConfigLoader.load_from_yaml(config_path)
        else:
            raise ValueError(f"Unsupported config format: {config_path}")
        
        return TrainingConfig(**config_dict)
    
    @staticmethod
    def save_config(config: BaseModel, output_path: str) -> None:
        """Save configuration to file."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = config.model_dump()
        
        if output_path.endswith('.json'):
            with open(output_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
        elif output_path.endswith(('.yaml', '.yml')):
            with open(output_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported output format: {output_path}")


# Global settings instance
settings = AppSettings()
