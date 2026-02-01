"""Model monitoring and performance tracking."""
import logging
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime
import json
from pathlib import Path
from prometheus_client import Counter, Histogram, Gauge, generate_latest

logger = logging.getLogger(__name__)


# Prometheus metrics
prediction_counter = Counter(
    'ml_predictions_total',
    'Total number of predictions made',
    ['model_name', 'model_version']
)

prediction_latency = Histogram(
    'ml_prediction_latency_seconds',
    'Prediction latency in seconds',
    ['model_name']
)

model_accuracy = Gauge(
    'ml_model_accuracy',
    'Current model accuracy',
    ['model_name', 'model_version']
)

data_drift_score = Gauge(
    'ml_data_drift_score',
    'Data drift detection score',
    ['feature_name']
)


class ModelMonitor:
    """Monitor model performance and predictions."""
    
    def __init__(self, log_dir: str = "monitoring/logs"):
        """
        Initialize model monitor.
        
        Args:
            log_dir: Directory to store monitoring logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.predictions_log = self.log_dir / "predictions.jsonl"
        self.metrics_log = self.log_dir / "metrics.json"
        
        self.metrics_history = []
        self._load_metrics_history()
    
    def _load_metrics_history(self) -> None:
        """Load metrics history from file."""
        if self.metrics_log.exists():
            with open(self.metrics_log, 'r') as f:
                self.metrics_history = json.load(f)
    
    def _save_metrics_history(self) -> None:
        """Save metrics history to file."""
        with open(self.metrics_log, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
    
    def log_prediction(self, model_name: str, model_version: str,
                      input_data: Dict[str, Any], prediction: Any,
                      latency: float, metadata: Optional[Dict] = None) -> None:
        """
        Log a prediction event.
        
        Args:
            model_name: Name of the model
            model_version: Version of the model
            input_data: Input features
            prediction: Model prediction
            latency: Prediction latency in seconds
            metadata: Additional metadata
        """
        try:
            # Update Prometheus metrics
            prediction_counter.labels(
                model_name=model_name,
                model_version=model_version
            ).inc()
            
            prediction_latency.labels(model_name=model_name).observe(latency)
            
            # Log to file
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'model_name': model_name,
                'model_version': model_version,
                'input_data': input_data,
                'prediction': str(prediction),
                'latency': latency,
                'metadata': metadata or {}
            }
            
            with open(self.predictions_log, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
            
            logger.debug(f"Logged prediction for {model_name} v{model_version}")
            
        except Exception as e:
            logger.error(f"Failed to log prediction: {e}")
    
    def log_metrics(self, model_name: str, model_version: str,
                   metrics: Dict[str, float]) -> None:
        """
        Log model performance metrics.
        
        Args:
            model_name: Name of the model
            model_version: Version of the model
            metrics: Performance metrics
        """
        try:
            # Update Prometheus metrics
            if 'accuracy' in metrics:
                model_accuracy.labels(
                    model_name=model_name,
                    model_version=model_version
                ).set(metrics['accuracy'])
            
            # Add to history
            metrics_entry = {
                'timestamp': datetime.now().isoformat(),
                'model_name': model_name,
                'model_version': model_version,
                'metrics': metrics
            }
            
            self.metrics_history.append(metrics_entry)
            self._save_metrics_history()
            
            logger.info(f"Logged metrics for {model_name} v{model_version}: {metrics}")
            
        except Exception as e:
            logger.error(f"Failed to log metrics: {e}")
    
    def get_prediction_stats(self, model_name: str,
                           time_window_hours: int = 24) -> Dict[str, Any]:
        """
        Get prediction statistics for a time window.
        
        Args:
            model_name: Name of the model
            time_window_hours: Time window in hours
            
        Returns:
            Statistics dictionary
        """
        try:
            from datetime import timedelta
            
            cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
            
            # Read predictions log
            predictions = []
            if self.predictions_log.exists():
                with open(self.predictions_log, 'r') as f:
                    for line in f:
                        pred = json.loads(line)
                        pred_time = datetime.fromisoformat(pred['timestamp'])
                        if pred['model_name'] == model_name and pred_time >= cutoff_time:
                            predictions.append(pred)
            
            if not predictions:
                return {'count': 0, 'avg_latency': 0}
            
            latencies = [p['latency'] for p in predictions]
            
            stats = {
                'count': len(predictions),
                'avg_latency': np.mean(latencies),
                'p50_latency': np.percentile(latencies, 50),
                'p95_latency': np.percentile(latencies, 95),
                'p99_latency': np.percentile(latencies, 99),
                'max_latency': np.max(latencies),
                'min_latency': np.min(latencies)
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get prediction stats: {e}")
            return {}
    
    def get_metrics_prometheus(self) -> bytes:
        """
        Get Prometheus metrics in exposition format.
        
        Returns:
            Metrics in Prometheus format
        """
        return generate_latest()
