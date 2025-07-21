"""
Feature engineering utilities for taxi data.
"""

import pandas as pd
from typing import List, Dict, Any


# Feature definitions from the notebook
NUMERIC_FEATURES = [
    "pickup_weekday",
    "pickup_hour", 
    "work_hours",
    "pickup_minute",
    "passenger_count",
    "trip_distance",
    "trip_time",
    "trip_speed"
]

CATEGORICAL_FEATURES = [
    "PULocationID",
    "DOLocationID", 
    "RatecodeID"
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Constants
EPS = 1e-7  # Small epsilon value to avoid division by zero


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create time-based features from pickup datetime.
    
    Args:
        df: Input DataFrame with taxi data
        
    Returns:
        DataFrame with time features added
    """
    df_copy = df.copy()
    
    # Extract time components
    df_copy['pickup_weekday'] = df_copy['tpep_pickup_datetime'].dt.weekday
    df_copy['pickup_hour'] = df_copy['tpep_pickup_datetime'].dt.hour
    df_copy['pickup_minute'] = df_copy['tpep_pickup_datetime'].dt.minute
    
    # Create work hours indicator (weekday 8am-6pm)
    df_copy['work_hours'] = (
        (df_copy['pickup_weekday'] >= 0) & 
        (df_copy['pickup_weekday'] <= 4) & 
        (df_copy['pickup_hour'] >= 8) & 
        (df_copy['pickup_hour'] <= 18)
    ).astype(int)
    
    return df_copy


def create_trip_features(df: pd.DataFrame, eps: float = EPS) -> pd.DataFrame:
    """
    Create trip-related features.
    
    Args:
        df: Input DataFrame with taxi data
        eps: Small epsilon value to avoid division by zero
        
    Returns:
        DataFrame with trip features added
    """
    df_copy = df.copy()
    
    # Calculate trip duration in seconds
    df_copy['trip_time'] = (
        df_copy['tpep_dropoff_datetime'] - df_copy['tpep_pickup_datetime']
    ).dt.total_seconds()
    
    # Calculate trip speed (miles per second, then can be converted)
    df_copy['trip_speed'] = df_copy['trip_distance'] / (df_copy['trip_time'] + eps)
    
    return df_copy


def create_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create all engineered features for the dataset.
    
    Args:
        df: Input DataFrame with taxi data
        
    Returns:
        DataFrame with all features added
    """
    print("Creating time-based features...")
    df = create_time_features(df)
    
    print("Creating trip-related features...")
    df = create_trip_features(df)
    
    print("Feature engineering completed.")
    return df


def select_features(df: pd.DataFrame, 
                   feature_columns: List[str] = None) -> pd.DataFrame:
    """
    Select specific features from DataFrame.
    
    Args:
        df: Input DataFrame
        feature_columns: List of column names to select (default: ALL_FEATURES)
        
    Returns:
        DataFrame with selected features only
    """
    if feature_columns is None:
        feature_columns = ALL_FEATURES
    
    # Check that all requested features exist
    missing_features = [f for f in feature_columns if f not in df.columns]
    if missing_features:
        raise ValueError(f"Missing features in DataFrame: {missing_features}")
    
    return df[feature_columns]


def get_feature_info() -> Dict[str, Any]:
    """
    Get information about available features.
    
    Returns:
        Dictionary with feature information
    """
    return {
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "all_features": ALL_FEATURES,
        "total_features": len(ALL_FEATURES),
        "eps_value": EPS
    }


def validate_features(df: pd.DataFrame, required_features: List[str] = None) -> bool:
    """
    Validate that DataFrame contains required features.
    
    Args:
        df: DataFrame to validate
        required_features: List of required feature names (default: ALL_FEATURES)
        
    Returns:
        True if all required features are present, False otherwise
    """
    if required_features is None:
        required_features = ALL_FEATURES
    
    missing_features = [f for f in required_features if f not in df.columns]
    
    if missing_features:
        print(f"Missing required features: {missing_features}")
        return False
    
    print("All required features are present.")
    return True
