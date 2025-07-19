"""
Data processing utilities for NYC taxi data.
"""

import pandas as pd
from typing import Optional, Tuple


def load_taxi_data(file_path: str) -> pd.DataFrame:
    """
    Load taxi data from a file.
    
    Args:
        file_path: Path to the data file
        
    Returns:
        Loaded DataFrame
    """
    return pd.read_parquet(file_path)


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess taxi data for modeling.
    
    Args:
        df: Raw taxi data DataFrame
        
    Returns:
        Preprocessed DataFrame
    """
    # Basic preprocessing steps will be implemented here
    return df


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
