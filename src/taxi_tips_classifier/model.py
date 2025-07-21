"""
Main model class and end-to-end pipeline for taxi tip prediction.
"""

import joblib
import pandas as pd
from typing import Any, Optional, Dict, List, Union
from pathlib import Path

from .data import load_taxi_data, preprocess_data
from .features import create_all_features, ALL_FEATURES, get_feature_info
from .train import create_model, train_model, save_model, load_model
from .predict import predict_single_trip, predict_batch, predict_and_evaluate


class TaxiTipClassifier:
    """
    Main class for taxi tip classification pipeline.
    """
    
    def __init__(self, model_params: Optional[Dict[str, Any]] = None):
        """
        Initialize the classifier.
        
        Args:
            model_params: Parameters for the RandomForest model
        """
        self.model_params = model_params or {
            "n_estimators": 100, 
            "max_depth": 10, 
            "random_state": 42
        }
        self.model = None
        self.features = ALL_FEATURES
        self.target_col = "high_tip"
        self.is_trained = False
        self.training_info = None
    
    def load_and_preprocess_data(self, 
                                data_source: str,
                                target_col: str = "high_tip") -> pd.DataFrame:
        """
        Load and preprocess data from URL or file path.
        
        Args:
            data_source: URL or file path to parquet data
            target_col: Name of target column
            
        Returns:
            Preprocessed DataFrame
        """
        print(f"Loading data from: {data_source}")
        
        # Load raw data
        raw_data = load_taxi_data(data_source)
        
        # Create features
        data_with_features = create_all_features(raw_data)
        
        # Preprocess
        processed_data = preprocess_data(
            data_with_features, 
            target_col=target_col,
            features=self.features
        )
        
        return processed_data
    
    def train(self, 
              training_data: Union[str, pd.DataFrame],
              sample_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            training_data: Path/URL to training data or DataFrame
            sample_size: Optional sample size for faster training
            
        Returns:
            Training information dictionary
        """
        # Load and preprocess data if path provided
        if isinstance(training_data, str):
            train_df = self.load_and_preprocess_data(training_data, self.target_col)
        else:
            train_df = training_data
        
        # Sample data if requested
        if sample_size is not None and len(train_df) > sample_size:
            print(f"Sampling {sample_size} records from {len(train_df)} total")
            train_df = train_df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        
        # Prepare features and target
        X_train = train_df[self.features]
        y_train = train_df[self.target_col]
        
        # Create and train model
        self.model = create_model(**self.model_params)
        self.model, self.training_info = train_model(
            self.model, X_train, y_train, self.features
        )
        
        self.is_trained = True
        print("Model training completed successfully!")
        
        return self.training_info
    
    def predict(self, 
                data: Union[str, pd.DataFrame, pd.Series],
                return_proba: bool = True) -> Union[float, int, pd.Series]:
        """
        Make predictions on new data.
        
        Args:
            data: Data to predict on (path, DataFrame, or single trip Series)
            return_proba: Whether to return probabilities or binary predictions
            
        Returns:
            Predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Handle different input types
        if isinstance(data, str):
            # Load and preprocess data from path/URL
            pred_data = self.load_and_preprocess_data(data, self.target_col)
            return predict_batch(self.model, pred_data, self.features, return_proba)
        
        elif isinstance(data, pd.DataFrame):
            # Batch prediction on DataFrame
            return predict_batch(self.model, data, self.features, return_proba)
        
        elif isinstance(data, pd.Series):
            # Single prediction
            return predict_single_trip(self.model, data, self.features, return_proba)
        
        else:
            raise ValueError("Data must be a file path, DataFrame, or Series")
    
    def evaluate(self, 
                 test_data: Union[str, pd.DataFrame],
                 threshold: float = 0.5) -> Dict[str, Any]:
        """
        Evaluate model performance on test data.
        
        Args:
            test_data: Test data (path/URL or DataFrame)
            threshold: Classification threshold
            
        Returns:
            Evaluation results dictionary
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        # Load and preprocess test data if path provided
        if isinstance(test_data, str):
            test_df = self.load_and_preprocess_data(test_data, self.target_col)
        else:
            test_df = test_data
        
        # Evaluate
        results = predict_and_evaluate(
            self.model, test_df, self.features, self.target_col, threshold
        )
        
        return results
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model and metadata.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        # Create model info
        model_info = {
            "model_params": self.model_params,
            "features": self.features,
            "target_col": self.target_col,
            "training_info": self.training_info
        }
        
        # Save model and info
        save_model(self.model, filepath, model_info)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
        """
        # Load model
        self.model = load_model(filepath)
        
        # Try to load model info
        info_filepath = filepath.replace('.joblib', '_info.joblib')
        try:
            model_info = joblib.load(info_filepath)
            self.model_params = model_info.get("model_params", self.model_params)
            self.features = model_info.get("features", self.features)
            self.target_col = model_info.get("target_col", self.target_col)
            self.training_info = model_info.get("training_info", None)
            print(f"Model and metadata loaded from {filepath}")
        except:
            print(f"Model loaded from {filepath} (metadata not found)")
        
        self.is_trained = True
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from trained model.
        
        Returns:
            DataFrame with features and their importance scores
        """
        if not self.is_trained:
            raise ValueError("Model must be trained to get feature importance")
        
        importance_df = pd.DataFrame({
            'feature': self.features,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def summary(self) -> Dict[str, Any]:
        """
        Get summary information about the classifier.
        
        Returns:
            Dictionary with classifier summary
        """
        summary_info = {
            "is_trained": self.is_trained,
            "model_params": self.model_params,
            "target_column": self.target_col,
            "num_features": len(self.features),
            "features": self.features
        }
        
        if self.training_info:
            summary_info["training_info"] = self.training_info
        
        return summary_info


# Convenience functions for quick usage
def train_taxi_tip_model(train_data_path: str,
                        sample_size: Optional[int] = None,
                        model_params: Optional[Dict[str, Any]] = None,
                        save_path: Optional[str] = None) -> TaxiTipClassifier:
    """
    Quick function to train a taxi tip classifier.
    
    Args:
        train_data_path: Path or URL to training data
        sample_size: Optional sample size for faster training
        model_params: Optional model parameters
        save_path: Optional path to save trained model
        
    Returns:
        Trained TaxiTipClassifier
    """
    classifier = TaxiTipClassifier(model_params)
    classifier.train(train_data_path, sample_size)
    
    if save_path:
        classifier.save_model(save_path)
    
    return classifier


def evaluate_model_on_data(model_path: str,
                          test_data_path: str) -> Dict[str, Any]:
    """
    Quick function to evaluate a saved model on test data.
    
    Args:
        model_path: Path to saved model
        test_data_path: Path or URL to test data
        
    Returns:
        Evaluation results
    """
    classifier = TaxiTipClassifier()
    classifier.load_model(model_path)
    results = classifier.evaluate(test_data_path)
    
    return results
