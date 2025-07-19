"""
Feature engineering utilities for taxi data.
"""

import pandas as pd
from typing import List


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features for taxi tip prediction.
    
    Args:
        df: Input DataFrame with taxi data
        
    Returns:
        DataFrame with engineered features
    """
    # Feature engineering will be implemented here
    return df


def select_features(df: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
    """
    Select specific features from DataFrame.
    
    Args:
        df: Input DataFrame
        feature_columns: List of column names to select
        
    Returns:
        DataFrame with selected features
    """
    return df[feature_columns]
