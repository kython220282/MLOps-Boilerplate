"""Model registry for managing model versions and metadata."""
import logging
import json
import shutil
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime
import joblib

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Local model registry for version control and metadata management."""
    
    def __init__(self, registry_path: str = "model_registry"):
        """
        Initialize model registry.
        
        Args:
            registry_path: Path to the registry directory
        """
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self.index_file = self.registry_path / "registry_index.json"
        
        self._load_index()
    
    def _load_index(self) -> None:
        """Load registry index from file."""
        if self.index_file.exists():
            with open(self.index_file, 'r') as f:
                self.index = json.load(f)
        else:
            self.index = {"models": {}}
            self._save_index()
    
    def _save_index(self) -> None:
        """Save registry index to file."""
        with open(self.index_file, 'w') as f:
            json.dump(self.index, f, indent=2)
    
    def register_model(self, model_name: str, model_path: str,
                      metadata: Dict[str, Any],
                      tags: Optional[Dict[str, str]] = None) -> str:
        """
        Register a new model version.
        
        Args:
            model_name: Name of the model
            model_path: Path to the model file
            metadata: Model metadata (metrics, config, etc.)
            tags: Optional tags
            
        Returns:
            Model version string
        """
        try:
            # Create model directory if it doesn't exist
            model_dir = self.registry_path / model_name
            model_dir.mkdir(exist_ok=True)
            
            # Determine version number
            if model_name in self.index["models"]:
                versions = self.index["models"][model_name]["versions"]
                version_num = max([int(v.split('_')[1]) for v in versions.keys()]) + 1
            else:
                version_num = 1
                self.index["models"][model_name] = {"versions": {}}
            
            version = f"v_{version_num}"
            
            # Create version directory
            version_dir = model_dir / version
            version_dir.mkdir(exist_ok=True)
            
            # Copy model file
            model_filename = Path(model_path).name
            dest_path = version_dir / model_filename
            shutil.copy2(model_path, dest_path)
            
            # Save metadata
            metadata_path = version_dir / "metadata.json"
            full_metadata = {
                "model_name": model_name,
                "version": version,
                "registered_at": datetime.now().isoformat(),
                "model_path": str(dest_path),
                "tags": tags or {},
                **metadata
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(full_metadata, f, indent=2)
            
            # Update index
            self.index["models"][model_name]["versions"][version] = {
                "registered_at": full_metadata["registered_at"],
                "path": str(version_dir),
                "stage": "Development",
                "metrics": metadata.get("metrics", {}),
                "tags": tags or {}
            }
            
            self._save_index()
            
            logger.info(f"Registered {model_name} {version}")
            return version
            
        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            raise
    
    def get_model(self, model_name: str, version: Optional[str] = None,
                 stage: Optional[str] = None) -> Dict[str, Any]:
        """
        Get model by name and version or stage.
        
        Args:
            model_name: Name of the model
            version: Specific version (e.g., 'v_1')
            stage: Stage name ('Development', 'Staging', 'Production')
            
        Returns:
            Model information dictionary
        """
        if model_name not in self.index["models"]:
            raise ValueError(f"Model {model_name} not found in registry")
        
        versions = self.index["models"][model_name]["versions"]
        
        if version:
            if version not in versions:
                raise ValueError(f"Version {version} not found for {model_name}")
            version_info = versions[version]
        elif stage:
            # Find latest version in the specified stage
            matching_versions = [
                (v, info) for v, info in versions.items()
                if info.get("stage") == stage
            ]
            if not matching_versions:
                raise ValueError(f"No models in stage {stage} for {model_name}")
            # Get latest by registered_at
            version, version_info = max(
                matching_versions,
                key=lambda x: x[1]["registered_at"]
            )
        else:
            # Get latest version
            version = max(versions.keys(), key=lambda x: int(x.split('_')[1]))
            version_info = versions[version]
        
        # Load metadata
        metadata_path = Path(version_info["path"]) / "metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        return metadata
    
    def load_model(self, model_name: str, version: Optional[str] = None,
                  stage: Optional[str] = None) -> Any:
        """
        Load model from registry.
        
        Args:
            model_name: Name of the model
            version: Specific version
            stage: Stage name
            
        Returns:
            Loaded model
        """
        metadata = self.get_model(model_name, version, stage)
        model_path = metadata["model_path"]
        
        model = joblib.load(model_path)
        logger.info(f"Loaded {model_name} {metadata['version']} from registry")
        
        return model
    
    def set_model_stage(self, model_name: str, version: str, stage: str) -> None:
        """
        Set model stage.
        
        Args:
            model_name: Name of the model
            version: Version to update
            stage: New stage ('Development', 'Staging', 'Production', 'Archived')
        """
        if model_name not in self.index["models"]:
            raise ValueError(f"Model {model_name} not found")
        
        if version not in self.index["models"][model_name]["versions"]:
            raise ValueError(f"Version {version} not found for {model_name}")
        
        self.index["models"][model_name]["versions"][version]["stage"] = stage
        self._save_index()
        
        logger.info(f"Set {model_name} {version} to stage {stage}")
    
    def list_models(self) -> List[str]:
        """
        List all registered models.
        
        Returns:
            List of model names
        """
        return list(self.index["models"].keys())
    
    def list_versions(self, model_name: str) -> List[Dict[str, Any]]:
        """
        List all versions of a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            List of version information
        """
        if model_name not in self.index["models"]:
            raise ValueError(f"Model {model_name} not found")
        
        versions = self.index["models"][model_name]["versions"]
        return [
            {"version": v, **info}
            for v, info in versions.items()
        ]
    
    def delete_version(self, model_name: str, version: str) -> None:
        """
        Delete a model version.
        
        Args:
            model_name: Name of the model
            version: Version to delete
        """
        if model_name not in self.index["models"]:
            raise ValueError(f"Model {model_name} not found")
        
        versions = self.index["models"][model_name]["versions"]
        if version not in versions:
            raise ValueError(f"Version {version} not found")
        
        # Remove directory
        version_dir = Path(versions[version]["path"])
        if version_dir.exists():
            shutil.rmtree(version_dir)
        
        # Remove from index
        del self.index["models"][model_name]["versions"][version]
        self._save_index()
        
        logger.info(f"Deleted {model_name} {version}")
