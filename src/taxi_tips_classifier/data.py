"""
Data processing utilities for NYC taxi data.
"""

import pandas as pd
from typing import Optional, List, Tuple


def load_taxi_data(file_path_or_url: str) -> pd.DataFrame:
    """
    Load taxi data from a local file or URL.
    
    Args:
        file_path_or_url: Path to the parquet file or URL
        
    Returns:
        Loaded DataFrame
    """
    print(f"Loading data from: {file_path_or_url}")
    df = pd.read_parquet(file_path_or_url)
    print(f"Data loaded successfully. Shape: {df.shape}")
    return df


def create_target_variable(df: pd.DataFrame, target_col: str = "high_tip", threshold: float = 0.2) -> pd.DataFrame:
    """
    Create target variable for high tip classification.
    
    Args:
        df: DataFrame with taxi data
        target_col: Name for the target column
        threshold: Tip fraction threshold for high tip classification (default: 0.2 = 20%)
        
    Returns:
        DataFrame with target variable added
    """
    # Calculate tip fraction
    df['tip_fraction'] = df['tip_amount'] / df['fare_amount']
    
    # Create binary target variable
    df[target_col] = df['tip_fraction'] > threshold
    
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic data cleaning for taxi dataset.
    
    Args:
        df: Raw taxi data DataFrame
        
    Returns:
        Cleaned DataFrame
    """
    print(f"Data shape before cleaning: {df.shape}")
    
    # Remove rows with zero or negative fare amounts to avoid divide-by-zero
    df = df[df['fare_amount'] > 0].reset_index(drop=True)
    
    print(f"Data shape after cleaning: {df.shape}")
    return df


def preprocess_data(df: pd.DataFrame, 
                   target_col: str = "high_tip",
                   features: Optional[List[str]] = None,
                   eps: float = 1e-7) -> pd.DataFrame:
    """
    Complete preprocessing pipeline for taxi data.
    
    Args:
        df: Raw taxi data DataFrame
        target_col: Name for the target column
        features: List of feature column names to keep
        eps: Small epsilon value to avoid division by zero
        
    Returns:
        Preprocessed DataFrame ready for modeling
    """
    # Default features if none provided
    if features is None:
        numeric_feat = [
            "pickup_weekday", "pickup_hour", 'work_hours', "pickup_minute",
            "passenger_count", 'trip_distance', 'trip_time', 'trip_speed'
        ]
        categorical_feat = ["PULocationID", "DOLocationID", "RatecodeID"]
        features = numeric_feat + categorical_feat
    
    # Basic cleaning
    df = clean_data(df)
    
    # Create target variable
    df = create_target_variable(df, target_col)
    
    # Feature engineering will be handled by build_features.py
    # For now, we'll add the basic time features here
    df['pickup_weekday'] = df['tpep_pickup_datetime'].dt.weekday
    df['pickup_hour'] = df['tpep_pickup_datetime'].dt.hour
    df['pickup_minute'] = df['tpep_pickup_datetime'].dt.minute
    df['work_hours'] = (df['pickup_weekday'] >= 0) & (df['pickup_weekday'] <= 4) & \
                       (df['pickup_hour'] >= 8) & (df['pickup_hour'] <= 18)
    df['trip_time'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds()
    df['trip_speed'] = df['trip_distance'] / (df['trip_time'] + eps)
    
    # Select only required columns
    columns_to_keep = ['tpep_dropoff_datetime'] + features + [target_col]
    df = df[columns_to_keep]
    
    # Fill missing values and convert to appropriate data types
    df[features + [target_col]] = df[features + [target_col]].astype("float32").fillna(-1.0)
    df[target_col] = df[target_col].astype("int32")
    
    return df.reset_index(drop=True)


def split_data(df: pd.DataFrame, 
               test_size: float = 0.2,
               random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into training and testing sets.
    
    Args:
        df: DataFrame to split
        test_size: Proportion of data for testing
        random_state: Random state for reproducibility
        
    Returns:
        Tuple of (train_df, test_df)
    """
    from sklearn.model_selection import train_test_split
    return train_test_split(df, test_size=test_size, random_state=random_state)
