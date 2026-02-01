"""Unit tests for configuration management."""
import pytest
import json
import yaml
from pathlib import Path
from ml_service.config import (
    ModelConfig,
    DataProcessingConfig,
    TrainingConfig,
    ConfigLoader,
    settings
)


class TestModelConfig:
    """Test suite for ModelConfig."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = ModelConfig()
        
        assert config.type == "random_forest"
        assert config.n_estimators == 100
        assert config.random_state == 42
    
    def test_custom_values(self):
        """Test custom configuration values."""
        config = ModelConfig(
            type="xgboost",
            n_estimators=200,
            learning_rate=0.05
        )
        
        assert config.type == "xgboost"
        assert config.n_estimators == 200
        assert config.learning_rate == 0.05
    
    def test_validation(self):
        """Test configuration validation."""
        with pytest.raises(Exception):
            ModelConfig(n_estimators=0)  # Should fail validation


class TestConfigLoader:
    """Test suite for ConfigLoader."""
    
    def test_load_from_json(self, tmp_path):
        """Test loading configuration from JSON."""
        config_data = {
            "model": {"type": "random_forest", "n_estimators": 50},
            "target_column": "price"
        }
        
        config_file = tmp_path / "config.json"
        with open(config_file, 'w') as f:
            json.dump(config_data, f)
        
        loaded = ConfigLoader.load_from_json(str(config_file))
        
        assert loaded["target_column"] == "price"
        assert loaded["model"]["n_estimators"] == 50
    
    def test_load_from_yaml(self, tmp_path):
        """Test loading configuration from YAML."""
        config_data = {
            "model": {"type": "xgboost", "n_estimators": 100},
            "target_column": "sales"
        }
        
        config_file = tmp_path / "config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        loaded = ConfigLoader.load_from_yaml(str(config_file))
        
        assert loaded["target_column"] == "sales"
        assert loaded["model"]["type"] == "xgboost"
    
    def test_save_config(self, tmp_path):
        """Test saving configuration to file."""
        config = TrainingConfig(target_column="target_var")
        output_file = tmp_path / "output.json"
        
        ConfigLoader.save_config(config, str(output_file))
        
        assert output_file.exists()
        with open(output_file) as f:
            saved_data = json.load(f)
        
        assert saved_data["target_column"] == "target_var"


class TestAppSettings:
    """Test suite for AppSettings."""
    
    def test_default_settings(self):
        """Test default application settings."""
        assert settings.app_name == "ml_service"
        assert settings.environment in ["development", "production", "staging"]
