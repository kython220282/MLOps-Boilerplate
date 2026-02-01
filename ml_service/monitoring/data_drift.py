"""Data drift detection utilities."""
import logging
from typing import Dict, Any, Optional, List, Callable
import pandas as pd
import numpy as np
from scipy import stats
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.metrics import *

logger = logging.getLogger(__name__)


class DataDriftDetector:
    """Detect data drift in production data."""
    
    def __init__(self, reference_data: Optional[pd.DataFrame] = None,
                 drift_threshold: float = 0.1):
        """
        Initialize drift detector.
        
        Args:
            reference_data: Reference/training data
            drift_threshold: Threshold for drift detection
        """
        self.reference_data = reference_data
        self.drift_threshold = drift_threshold
        self.drift_scores = {}
    
    def set_reference_data(self, reference_data: pd.DataFrame) -> None:
        """
        Set reference data for drift detection.
        
        Args:
            reference_data: Reference DataFrame
        """
        self.reference_data = reference_data
        logger.info(f"Set reference data with shape {reference_data.shape}")
    
    def detect_feature_drift_ks(self, feature_name: str,
                                 current_data: pd.Series) -> Dict[str, Any]:
        """
        Detect drift using Kolmogorov-Smirnov test.
        
        Args:
            feature_name: Name of the feature
            current_data: Current production data
            
        Returns:
            Drift detection results
        """
        if self.reference_data is None:
            raise ValueError("Reference data not set")
        
        reference_values = self.reference_data[feature_name]
        
        # KS test - returns (statistic, pvalue) tuple in both old and new scipy
        ks_result = stats.ks_2samp(reference_values, current_data)
        # Access as tuple indices for compatibility
        statistic = ks_result[0]
        p_value = ks_result[1]
        
        is_drift = p_value < self.drift_threshold
        
        result = {
            'feature': feature_name,
            'statistic': float(statistic),
            'p_value': float(p_value),
            'is_drift': is_drift,
            'method': 'ks_test'
        }
        
        logger.info(f"Drift detection for {feature_name}: p_value={p_value:.4f}, drift={is_drift}")
        
        return result
    
    def detect_drift_psi(self, feature_name: str,
                        current_data: pd.Series,
                        n_bins: int = 10) -> Dict[str, Any]:
        """
        Detect drift using Population Stability Index (PSI).
        
        Args:
            feature_name: Name of the feature
            current_data: Current production data
            n_bins: Number of bins for discretization
            
        Returns:
            Drift detection results
        """
        if self.reference_data is None:
            raise ValueError("Reference data not set")
        
        reference_values = self.reference_data[feature_name]
        
        # Create bins
        bins = np.linspace(
            min(reference_values.min(), current_data.min()),
            max(reference_values.max(), current_data.max()),
            n_bins + 1
        )
        
        # Calculate distributions
        ref_dist = np.histogram(reference_values, bins=bins)[0] / len(reference_values)
        curr_dist = np.histogram(current_data, bins=bins)[0] / len(current_data)
        
        # Avoid division by zero
        ref_dist = np.where(ref_dist == 0, 0.0001, ref_dist)
        curr_dist = np.where(curr_dist == 0, 0.0001, curr_dist)
        
        # Calculate PSI
        psi = np.sum((curr_dist - ref_dist) * np.log(curr_dist / ref_dist))
        
        # Interpret PSI
        # PSI < 0.1: No significant change
        # 0.1 <= PSI < 0.25: Moderate change
        # PSI >= 0.25: Significant change
        is_drift = psi >= 0.25
        
        result = {
            'feature': feature_name,
            'psi': float(psi),
            'is_drift': is_drift,
            'method': 'psi'
        }
        
        logger.info(f"PSI for {feature_name}: {psi:.4f}, drift={is_drift}")
        
        return result
    
    def detect_all_features_drift(self, current_data: pd.DataFrame,
                                  method: str = 'ks') -> Dict[str, Dict[str, Any]]:
        """
        Detect drift for all features.
        
        Args:
            current_data: Current production data
            method: Detection method ('ks' or 'psi')
            
        Returns:
            Dictionary of drift results per feature
        """
        if self.reference_data is None:
            raise ValueError("Reference data not set")
        
        results = {}
        
        for column in current_data.columns:
            if column in self.reference_data.columns:
                try:
                    if method == 'ks':
                        result = self.detect_feature_drift_ks(column, current_data[column])
                    elif method == 'psi':
                        result = self.detect_drift_psi(column, current_data[column])
                    else:
                        raise ValueError(f"Unknown method: {method}")
                    
                    results[column] = result
                    
                except Exception as e:
                    logger.error(f"Failed to detect drift for {column}: {e}")
                    results[column] = {'error': str(e)}
        
        # Calculate overall drift
        drifted_features = sum(1 for r in results.values() if r.get('is_drift', False))
        results['_summary'] = {
            'total_features': len(results) - 1,
            'drifted_features': drifted_features,
            'drift_ratio': drifted_features / max(len(results) - 1, 1)
        }
        
        return results
    
    def generate_drift_report_evidently(self, current_data: pd.DataFrame,
                                       output_path: str = "drift_report.html") -> None:
        """
        Generate comprehensive drift report using Evidently.
        
        Args:
            current_data: Current production data
            output_path: Path to save HTML report
        """
        if self.reference_data is None:
            raise ValueError("Reference data not set")
        
        try:
            # Create report
            report = Report(metrics=[
                DataDriftPreset(),
                DataQualityPreset(),
            ])
            
            report.run(
                reference_data=self.reference_data,
                current_data=current_data
            )
            
            # Save report
            report.save_html(output_path)
            
            logger.info(f"Drift report saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to generate drift report: {e}")
            raise
    
    def get_drift_summary(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Get summary of data drift.
        
        Args:
            current_data: Current production data
            
        Returns:
            Drift summary
        """
        drift_results = self.detect_all_features_drift(current_data, method='ks')
        
        summary = drift_results.get('_summary', {})
        summary['timestamp'] = pd.Timestamp.now().isoformat()
        summary['current_data_size'] = len(current_data)
        summary['reference_data_size'] = len(self.reference_data) if self.reference_data is not None else 0
        
        return summary
