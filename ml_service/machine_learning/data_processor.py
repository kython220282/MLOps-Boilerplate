"""Data processing and feature engineering module."""
import logging
from typing import Optional, List, Dict, Any, Tuple
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split


logger = logging.getLogger(__name__)


class DataProcessor:
    """Handle data preprocessing and feature engineering."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.scalers = {}
        self.encoders = {}
        self.feature_columns = None
        self.target_column = None
    
    def load_data(self, data_source: Any) -> pd.DataFrame:
        """Load data from various sources."""
        try:
            if isinstance(data_source, str):
                # Load from file
                if data_source.endswith('.csv'):
                    df = pd.read_csv(data_source)
                elif data_source.endswith('.parquet'):
                    df = pd.read_parquet(data_source)
                else:
                    raise ValueError(f"Unsupported file format: {data_source}")
            elif isinstance(data_source, pd.DataFrame):
                df = data_source.copy()
            else:
                raise ValueError("Unsupported data source type")
            
            logger.info(f"Loaded data with shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise
    
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'mean') -> pd.DataFrame:
        """Handle missing values in the dataset."""
        try:
            df_processed = df.copy()
            
            if strategy == 'drop':
                df_processed = df_processed.dropna()
            elif strategy == 'mean':
                numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
                df_processed[numeric_cols] = df_processed[numeric_cols].fillna(df_processed[numeric_cols].mean())
            elif strategy == 'median':
                numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
                df_processed[numeric_cols] = df_processed[numeric_cols].fillna(df_processed[numeric_cols].median())
            elif strategy == 'mode':
                for col in df_processed.columns:
                    df_processed[col].fillna(df_processed[col].mode()[0], inplace=True)
            
            logger.info(f"Handled missing values using strategy: {strategy}")
            return df_processed
        except Exception as e:
            logger.error(f"Failed to handle missing values: {e}")
            raise
    
    def encode_categorical_features(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Encode categorical features."""
        try:
            df_processed = df.copy()
            
            if columns is None:
                columns = df_processed.select_dtypes(include=['object']).columns.tolist()
            
            for col in columns:
                if col in df_processed.columns:
                    if col not in self.encoders:
                        self.encoders[col] = LabelEncoder()
                        df_processed[col] = self.encoders[col].fit_transform(df_processed[col].astype(str))
                    else:
                        df_processed[col] = self.encoders[col].transform(df_processed[col].astype(str))
            
            logger.info(f"Encoded {len(columns)} categorical columns")
            return df_processed
        except Exception as e:
            logger.error(f"Failed to encode categorical features: {e}")
            raise
    
    def scale_features(self, df: pd.DataFrame, method: str = 'standard', 
                      columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Scale numerical features."""
        try:
            df_processed = df.copy()
            
            if columns is None:
                columns = df_processed.select_dtypes(include=[np.number]).columns.tolist()
            
            scaler_key = f"{method}_scaler"
            if scaler_key not in self.scalers:
                if method == 'standard':
                    self.scalers[scaler_key] = StandardScaler()
                elif method == 'minmax':
                    self.scalers[scaler_key] = MinMaxScaler()
                else:
                    raise ValueError(f"Unsupported scaling method: {method}")
                
                df_processed[columns] = self.scalers[scaler_key].fit_transform(df_processed[columns])
            else:
                df_processed[columns] = self.scalers[scaler_key].transform(df_processed[columns])
            
            logger.info(f"Scaled {len(columns)} columns using {method} scaling")
            return df_processed
        except Exception as e:
            logger.error(f"Failed to scale features: {e}")
            raise
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create additional features through feature engineering."""
        try:
            df_processed = df.copy()
            
            # Example feature engineering operations
            # Add your custom feature engineering logic here
            
            logger.info("Feature engineering completed")
            return df_processed
        except Exception as e:
            logger.error(f"Failed to create features: {e}")
            raise
    
    def split_data(self, df: pd.DataFrame, target_column: str, 
                   test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split data into training and testing sets."""
        try:
            self.target_column = target_column
            self.feature_columns = [col for col in df.columns if col != target_column]
            
            X = df[self.feature_columns]
            y = df[target_column]
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            logger.info(f"Split data - Train: {X_train.shape}, Test: {X_test.shape}")
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logger.error(f"Failed to split data: {e}")
            raise
    
    def preprocess_pipeline(self, df: pd.DataFrame, target_column: str, 
                          fit: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Complete preprocessing pipeline."""
        try:
            # Handle missing values
            df_processed = self.handle_missing_values(df, strategy='mean')
            
            # Encode categorical features
            df_processed = self.encode_categorical_features(df_processed)
            
            # Create additional features
            df_processed = self.create_features(df_processed)
            
            # Split data
            X_train, X_test, y_train, y_test = self.split_data(
                df_processed, target_column
            )
            
            # Scale features
            if fit:
                X_train = self.scale_features(X_train, method='standard')
            X_test = self.scale_features(X_test, method='standard')
            
            logger.info("Preprocessing pipeline completed successfully")
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logger.error(f"Preprocessing pipeline failed: {e}")
            raise
