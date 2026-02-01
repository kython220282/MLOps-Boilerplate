"""Inference application entry point."""
import logging
import argparse
import json
from pathlib import Path
import sys
import pandas as pd
import numpy as np
from typing import Union, List, Dict, Any, Optional

from ml_service.machine_learning.model import ModelFactory
from ml_service.machine_learning.data_processor import DataProcessor


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/inference.log')
    ]
)

logger = logging.getLogger(__name__)


class InferenceService:
    """Service for making predictions with trained models."""
    
    def __init__(self, model_path: str, metadata_path: Optional[str] = None):
        self.model_path = model_path
        self.metadata_path = metadata_path or model_path.replace('.joblib', '_metadata.json')
        self.model: Any = None
        self.data_processor: Optional[DataProcessor] = None
        self.metadata: Optional[Dict[str, Any]] = None
        
        self._load_model()
        self._load_metadata()
    
    def _load_model(self) -> None:
        """Load trained model."""
        try:
            # For now, create a generic model instance and load
            from ml_service.machine_learning.model import BaseModel
            import joblib
            
            self.model = joblib.load(self.model_path)
            logger.info(f"Model loaded from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _load_metadata(self) -> None:
        """Load model metadata."""
        try:
            if Path(self.metadata_path).exists():
                with open(self.metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                logger.info(f"Metadata loaded from {self.metadata_path}")
                
                # Initialize data processor with saved config
                if self.metadata:
                    processor_config = self.metadata.get('config', {}).get('data_processing', {})
                    self.data_processor = DataProcessor(processor_config)
                else:
                    self.data_processor = DataProcessor()
            else:
                logger.warning(f"Metadata file not found: {self.metadata_path}")
                self.data_processor = DataProcessor()
        except Exception as e:
            logger.error(f"Failed to load metadata: {e}")
            raise
    
    def preprocess_input(self, data: Union[pd.DataFrame, Dict, List]) -> pd.DataFrame:
        """Preprocess input data for prediction."""
        try:
            # Convert input to DataFrame
            if isinstance(data, dict):
                df = pd.DataFrame([data])
            elif isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, pd.DataFrame):
                df = data.copy()
            else:
                raise ValueError(f"Unsupported input type: {type(data)}")
            
            # Apply preprocessing if data processor is available
            if self.data_processor:
                # Handle missing values
                df = self.data_processor.handle_missing_values(df, strategy='mean')
                
                # Encode categorical features
                df = self.data_processor.encode_categorical_features(df)
                
                # Scale features
                df = self.data_processor.scale_features(df, method='standard')
            
            logger.info(f"Preprocessed input data with shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Input preprocessing failed: {e}")
            raise
    
    def predict(self, data: Union[pd.DataFrame, Dict, List]) -> np.ndarray:
        """Make predictions on input data."""
        try:
            # Preprocess input
            df = self.preprocess_input(data)
            
            # Make predictions
            predictions = self.model.predict(df)
            
            logger.info(f"Generated {len(predictions)} predictions")
            return predictions
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise
    
    def predict_proba(self, data: Union[pd.DataFrame, Dict, List]) -> np.ndarray:
        """Get prediction probabilities (for classification models)."""
        try:
            df = self.preprocess_input(data)
            
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(df)
                logger.info(f"Generated probability predictions for {len(probabilities)} samples")
                return probabilities
            else:
                raise AttributeError("Model does not support probability predictions")
        except Exception as e:
            logger.error(f"Probability prediction failed: {e}")
            raise
    
    def batch_predict(self, input_path: str, output_path: str) -> None:
        """Process batch predictions from file."""
        try:
            # Load input data
            if input_path.endswith('.csv'):
                df = pd.read_csv(input_path)
            elif input_path.endswith('.parquet'):
                df = pd.read_parquet(input_path)
            else:
                raise ValueError(f"Unsupported file format: {input_path}")
            
            # Make predictions
            predictions = self.predict(df)
            
            # Add predictions to DataFrame
            df['predictions'] = predictions
            
            # Save results
            if output_path.endswith('.csv'):
                df.to_csv(output_path, index=False)
            elif output_path.endswith('.parquet'):
                df.to_parquet(output_path, index=False)
            
            logger.info(f"Batch predictions saved to {output_path}")
        except Exception as e:
            logger.error(f"Batch prediction failed: {e}")
            raise


def main():
    """Main inference application."""
    parser = argparse.ArgumentParser(description='ML Model Inference Application')
    parser.add_argument(
        '--model-path',
        type=str,
        required=True,
        help='Path to trained model file'
    )
    parser.add_argument(
        '--input-path',
        type=str,
        required=True,
        help='Path to input data file'
    )
    parser.add_argument(
        '--output-path',
        type=str,
        required=True,
        help='Path to save predictions'
    )
    parser.add_argument(
        '--metadata-path',
        type=str,
        default=None,
        help='Path to model metadata file'
    )
    
    args = parser.parse_args()
    
    try:
        # Create logs directory
        Path('logs').mkdir(parents=True, exist_ok=True)
        
        # Initialize inference service
        logger.info("Initializing inference service...")
        service = InferenceService(args.model_path, args.metadata_path)
        
        # Run batch predictions
        logger.info(f"Processing predictions for {args.input_path}...")
        service.batch_predict(args.input_path, args.output_path)
        
        logger.info(f"Inference completed successfully!")
        logger.info(f"Predictions saved to {args.output_path}")
        
        return 0
    
    except Exception as e:
        logger.error(f"Inference application failed: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
