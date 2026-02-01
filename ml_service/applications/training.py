"""Training application entry point."""
import logging
import argparse
import json
from pathlib import Path
import sys

from ml_service.machine_learning.training_pipeline import TrainingPipeline


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/training.log')
    ]
)

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from JSON file."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        raise


def main():
    """Main training application."""
    parser = argparse.ArgumentParser(description='ML Model Training Application')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to configuration JSON file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='models',
        help='Directory to save trained models'
    )
    
    args = parser.parse_args()
    
    try:
        # Create output directory
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        Path('logs').mkdir(parents=True, exist_ok=True)
        
        # Load configuration
        config = load_config(args.config)
        
        # Update output path in config
        if 'output_path' not in config:
            config['output_path'] = f"{args.output_dir}/model.joblib"
        
        # Initialize and run training pipeline
        logger.info("Initializing training pipeline...")
        pipeline = TrainingPipeline(config)
        
        metrics = pipeline.run_pipeline()
        
        # Save results
        results_path = f"{args.output_dir}/training_results.json"
        with open(results_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Training completed successfully!")
        logger.info(f"Results saved to {results_path}")
        logger.info(f"Model saved to {metrics.get('model_path')}")
        
        return 0
    
    except Exception as e:
        logger.error(f"Training application failed: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
