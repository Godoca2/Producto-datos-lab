"""
Prediction utilities for taxi tip classification.
"""

import joblib
import pandas as pd
import numpy as np
from typing import Union, List, Dict, Any, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report, confusion_matrix, accuracy_score, precision_score, recall_score


def load_trained_model(model_path: str) -> RandomForestClassifier:
    """
    Load a pre-trained model from disk.
    
    Args:
        model_path: Path to the saved model file
        
    Returns:
        Loaded RandomForestClassifier model
    """
    try:
        model = joblib.load(model_path)
        print(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        raise


def predict_single_trip(model: RandomForestClassifier,
                       trip_features: Union[pd.Series, np.ndarray, List],
                       features: List[str],
                       return_proba: bool = True) -> Union[float, int]:
    """
    Predict tip amount for a single trip.
    
    Args:
        model: Trained RandomForestClassifier
        trip_features: Features for a single trip
        features: List of feature names in correct order
        return_proba: If True, return probability of high tip; if False, return binary prediction
        
    Returns:
        Prediction (probability or binary class)
    """
    # Ensure input is in correct format
    if isinstance(trip_features, pd.Series):
        trip_features = trip_features[features].values
    elif isinstance(trip_features, list):
        trip_features = np.array(trip_features)
    
    # Reshape for sklearn (single sample)
    trip_features = trip_features.reshape(1, -1)
    
    if return_proba:
        # Return probability of positive class (high tip)
        proba = model.predict_proba(trip_features)[0][1]
        return proba
    else:
        # Return binary prediction
        prediction = model.predict(trip_features)[0]
        return int(prediction)


def predict_batch(model: RandomForestClassifier,
                  X: pd.DataFrame,
                  features: List[str],
                  return_proba: bool = False) -> np.ndarray:
    """
    Make predictions for a batch of trips.
    
    Args:
        model: Trained RandomForestClassifier
        X: DataFrame with trip features
        features: List of feature names to use
        return_proba: If True, return probabilities; if False, return binary predictions
        
    Returns:
        Array of predictions
    """
    X_selected = X[features]
    
    if return_proba:
        # Return probabilities for positive class
        probabilities = model.predict_proba(X_selected)[:, 1]
        return probabilities
    else:
        # Return binary predictions
        predictions = model.predict(X_selected)
        return predictions


def evaluate_predictions(y_true: Union[pd.Series, np.ndarray],
                        y_pred: Union[pd.Series, np.ndarray],
                        y_pred_proba: Optional[Union[pd.Series, np.ndarray]] = None) -> Dict[str, Any]:
    """
    Comprehensive evaluation of model predictions.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities (optional)
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Convert to numpy arrays if needed
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    if isinstance(y_pred, pd.Series):
        y_pred = y_pred.values
    
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary')
    recall = recall_score(y_true, y_pred, average='binary')
    f1 = f1_score(y_true, y_pred, average='binary')
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Classification report
    report = classification_report(y_true, y_pred, output_dict=True)
    
    # Compile results
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'support': {
            'total_samples': len(y_true),
            'positive_samples': int(np.sum(y_true)),
            'negative_samples': int(len(y_true) - np.sum(y_true))
        }
    }
    
    print(f"Evaluation Results:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    
    return results


def predict_and_evaluate(model: RandomForestClassifier,
                         test_data: pd.DataFrame,
                         features: List[str],
                         target_col: str = "high_tip",
                         threshold: float = 0.5) -> Dict[str, Any]:
    """
    Complete prediction and evaluation pipeline.
    
    Args:
        model: Trained RandomForestClassifier
        test_data: Test dataset
        features: List of feature names
        target_col: Name of target column
        threshold: Probability threshold for binary classification
        
    Returns:
        Dictionary with predictions and evaluation results
    """
    print(f"Making predictions on {len(test_data)} samples...")
    
    # Prepare data
    X_test = test_data[features]
    y_test = test_data[target_col]
    
    # Get probabilities and binary predictions
    y_pred_proba = predict_batch(model, test_data, features, return_proba=True)
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Evaluate predictions
    evaluation_results = evaluate_predictions(y_test, y_pred, y_pred_proba)
    
    # Add threshold info
    evaluation_results['threshold_used'] = threshold
    evaluation_results['predictions'] = {
        'probabilities': y_pred_proba.tolist(),
        'binary_predictions': y_pred.tolist(),
        'true_labels': y_test.tolist()
    }
    
    return evaluation_results


def compare_datasets(model: RandomForestClassifier,
                     datasets: Dict[str, pd.DataFrame],
                     features: List[str],
                     target_col: str = "high_tip") -> Dict[str, Dict[str, Any]]:
    """
    Compare model performance across multiple datasets.
    
    Args:
        model: Trained RandomForestClassifier
        datasets: Dictionary of dataset name -> DataFrame pairs
        features: List of feature names
        target_col: Name of target column
        
    Returns:
        Dictionary with results for each dataset
    """
    results = {}
    
    for dataset_name, dataset in datasets.items():
        print(f"\nEvaluating on {dataset_name} dataset...")
        dataset_results = predict_and_evaluate(model, dataset, features, target_col)
        results[dataset_name] = dataset_results
    
    # Print comparison summary
    print(f"\n{'Dataset':<15} {'F1-Score':<10} {'Accuracy':<10} {'Samples':<10}")
    print("-" * 50)
    for name, result in results.items():
        f1 = result['f1_score']
        acc = result['accuracy']
        samples = result['support']['total_samples']
        print(f"{name:<15} {f1:<10.4f} {acc:<10.4f} {samples:<10}")
    
    return results


def save_predictions(predictions: Dict[str, Any], 
                    filepath: str) -> None:
    """
    Save predictions and evaluation results to disk.
    
    Args:
        predictions: Dictionary with prediction results
        filepath: Path to save the results
    """
    try:
        joblib.dump(predictions, filepath)
        print(f"Predictions saved to {filepath}")
    except Exception as e:
        print(f"Error saving predictions: {e}")
        raise
