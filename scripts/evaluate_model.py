#!/usr/bin/env python3
"""
Evaluation script for taxi tip classifier.

Usage:
    python scripts/evaluate_model.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from taxi_tips_classifier.model import evaluate_model_on_data


def main():
    """Evaluate a trained taxi tip classifier."""
    
    print("Evaluating NYC Taxi Tip Classifier...")
    
    # Model and test data paths
    model_path = "models/simple_taxi_model.joblib"
    test_data_url = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2020-02.parquet"
    
    try:
        # Evaluate model
        results = evaluate_model_on_data(model_path, test_data_url)
        
        print("\nEvaluation Results:")
        print(f"- F1 Score: {results['f1_score']:.4f}")
        print(f"- Accuracy: {results['accuracy']:.4f}")
        print(f"- Precision: {results['precision']:.4f}")
        print(f"- Recall: {results['recall']:.4f}")
        
        print(f"\nDataset Info:")
        print(f"- Total samples: {results['support']['total_samples']}")
        print(f"- Positive samples: {results['support']['positive_samples']}")
        print(f"- Negative samples: {results['support']['negative_samples']}")
        
        print("\nConfusion Matrix:")
        cm = results['confusion_matrix']
        print(f"                Predicted")
        print(f"               Low   High")
        print(f"Actual Low   {cm[0][0]:>5} {cm[0][1]:>5}")
        print(f"      High   {cm[1][0]:>5} {cm[1][1]:>5}")
        
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        print("Please run train_simple.py first to create the model.")
    except Exception as e:
        print(f"Error during evaluation: {e}")


if __name__ == "__main__":
    main()
