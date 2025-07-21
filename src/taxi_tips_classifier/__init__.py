"""
NYC Taxi Tips Classifier Package
================================

A machine learning package for predicting taxi tip amounts in New York City.

Main Components:
- data: Data loading and preprocessing utilities
- features: Feature engineering functions  
- train: Model training utilities
- predict: Prediction and evaluation functions
- model: Main classifier class and end-to-end pipeline
- plots: Visualization functions

Quick Start:
-----------
>>> from taxi_tips_classifier.model import TaxiTipClassifier
>>> classifier = TaxiTipClassifier()
>>> classifier.train("path/to/training_data.parquet")
>>> results = classifier.evaluate("path/to/test_data.parquet")
>>> print(f"F1 Score: {results['f1_score']:.4f}")

>>> # Or use convenience functions
>>> from taxi_tips_classifier.model import train_taxi_tip_model
>>> classifier = train_taxi_tip_model("path/to/data.parquet", save_path="model.joblib")
"""

__version__ = "0.1.0"
__author__ = "César Godoy Delaigue"
__email__ = "cesar.delaigue@gmail.com"

# Import main classes and functions for easy access
from .model import TaxiTipClassifier, train_taxi_tip_model, evaluate_model_on_data
from .features import ALL_FEATURES, NUMERIC_FEATURES, CATEGORICAL_FEATURES, get_feature_info
from .data import load_taxi_data, preprocess_data, clean_data, create_target_variable
from .temporal_evaluation import TemporalModelEvaluator, evaluate_model_temporal_performance

# Define what gets imported with "from taxi_tips_classifier import *"
__all__ = [
    # Main classes
    "TaxiTipClassifier",
    "TemporalModelEvaluator",
    
    # Convenience functions
    "train_taxi_tip_model",
    "evaluate_model_on_data",
    "evaluate_model_temporal_performance",
    
    # Data functions
    "load_taxi_data",
    "preprocess_data", 
    "clean_data",
    "create_target_variable",
    
    # Feature information
    "ALL_FEATURES",
    "NUMERIC_FEATURES", 
    "CATEGORICAL_FEATURES",
    "get_feature_info",
    
    # Package metadata
    "__version__",
    "__author__",
    "__email__"
]
