"""
Model training utilities for taxi tip prediction.
"""

import joblib
import pandas as pd
from typing import Optional, Dict, Any, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import time


def create_model(n_estimators: int = 100, 
                max_depth: int = 10,
                random_state: int = 42,
                **kwargs) -> RandomForestClassifier:
    """
    Create a RandomForest classifier with specified parameters.
    
    Args:
        n_estimators: Number of trees in the forest
        max_depth: Maximum depth of the trees
        random_state: Random state for reproducibility
        **kwargs: Additional parameters for RandomForestClassifier
        
    Returns:
        Initialized RandomForestClassifier
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        **kwargs
    )
    
    return model


def train_model(model: RandomForestClassifier,
                X_train: pd.DataFrame,
                y_train: pd.Series,
                features: Optional[list] = None) -> Tuple[RandomForestClassifier, Dict[str, Any]]:
    """
    Train the RandomForest model.
    
    Args:
        model: RandomForestClassifier instance
        X_train: Training features
        y_train: Training targets
        features: List of feature names to use (if None, use all columns in X_train)
        
    Returns:
        Tuple of (trained_model, training_info)
    """
    # Select features if specified
    if features is not None:
        X_train_selected = X_train[features]
    else:
        X_train_selected = X_train
        features = list(X_train.columns)
    
    print(f"Training model with {len(features)} features...")
    print(f"Training data shape: {X_train_selected.shape}")
    print(f"Target distribution: {y_train.value_counts().to_dict()}")
    
    # Track training time
    start_time = time.time()
    
    # Train the model
    model.fit(X_train_selected, y_train)
    
    end_time = time.time()
    training_time = end_time - start_time
    
    print(f"Model training completed in {training_time:.2f} seconds")
    
    # Gather training information
    training_info = {
        "training_time": training_time,
        "n_samples": len(X_train_selected),
        "n_features": len(features),
        "features_used": features,
        "target_distribution": y_train.value_counts().to_dict(),
        "model_params": model.get_params()
    }
    
    return model, training_info


def evaluate_model(model: RandomForestClassifier,
                  X_test: pd.DataFrame,
                  y_test: pd.Series,
                  features: Optional[list] = None) -> Dict[str, Any]:
    """
    Evaluate the trained model on test data.
    
    Args:
        model: Trained RandomForestClassifier
        X_test: Test features
        y_test: Test targets
        features: List of feature names to use
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Select features if specified
    if features is not None:
        X_test_selected = X_test[features]
    else:
        X_test_selected = X_test
    
    print(f"Evaluating model on test data...")
    print(f"Test data shape: {X_test_selected.shape}")
    
    # Make predictions
    start_time = time.time()
    y_pred_proba = model.predict_proba(X_test_selected)
    y_pred = model.predict(X_test_selected)
    prediction_time = time.time() - start_time
    
    # Extract probability for positive class (high tip)
    y_pred_proba_positive = y_pred_proba[:, 1]
    
    # Calculate metrics
    f1 = f1_score(y_test, y_pred)
    
    # Get classification report as dict
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    
    # Get confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    evaluation_results = {
        "f1_score": f1,
        "prediction_time": prediction_time,
        "classification_report": report_dict,
        "confusion_matrix": cm.tolist(),  # Convert to list for JSON serialization
        "n_test_samples": len(X_test_selected),
        "test_target_distribution": y_test.value_counts().to_dict()
    }
    
    print(f"F1 Score: {f1:.4f}")
    print(f"Prediction completed in {prediction_time:.2f} seconds")
    
    return evaluation_results


def save_model(model: RandomForestClassifier, 
               filepath: str,
               training_info: Optional[Dict[str, Any]] = None) -> None:
    """
    Save the trained model to disk.
    
    Args:
        model: Trained model to save
        filepath: Path where to save the model
        training_info: Optional training information to save alongside
    """
    try:
        # Save the model
        joblib.dump(model, filepath)
        
        # Save training info if provided
        if training_info is not None:
            info_filepath = filepath.replace('.joblib', '_info.joblib')
            joblib.dump(training_info, info_filepath)
            print(f"Model and training info saved to {filepath} and {info_filepath}")
        else:
            print(f"Model saved to {filepath}")
            
    except Exception as e:
        print(f"Error saving model: {e}")
        raise


def load_model(filepath: str) -> RandomForestClassifier:
    """
    Load a trained model from disk.
    
    Args:
        filepath: Path to the saved model
        
    Returns:
        Loaded RandomForestClassifier
    """
    try:
        model = joblib.load(filepath)
        print(f"Model loaded from {filepath}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        raise


def full_training_pipeline(train_data: pd.DataFrame,
                          features: list,
                          target_col: str = "high_tip",
                          test_data: Optional[pd.DataFrame] = None,
                          model_params: Optional[Dict[str, Any]] = None,
                          save_path: Optional[str] = None) -> Tuple[RandomForestClassifier, Dict[str, Any]]:
    """
    Complete training pipeline from data to trained model.
    
    Args:
        train_data: Training dataset
        features: List of feature column names
        target_col: Name of target column
        test_data: Optional test dataset for evaluation
        model_params: Optional model parameters
        save_path: Optional path to save the trained model
        
    Returns:
        Tuple of (trained_model, results_dict)
    """
    if model_params is None:
        model_params = {"n_estimators": 100, "max_depth": 10, "random_state": 42}
    
    # Prepare data
    X_train = train_data[features]
    y_train = train_data[target_col]
    
    # Create and train model
    model = create_model(**model_params)
    trained_model, training_info = train_model(model, X_train, y_train, features)
    
    results = {"training_info": training_info}
    
    # Evaluate on test data if provided
    if test_data is not None:
        X_test = test_data[features]
        y_test = test_data[target_col]
        evaluation_results = evaluate_model(trained_model, X_test, y_test, features)
        results["evaluation_results"] = evaluation_results
    
    # Save model if path provided
    if save_path is not None:
        save_model(trained_model, save_path, training_info)
    
    return trained_model, results
