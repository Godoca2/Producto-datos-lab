#!/usr/bin/env python3
"""
Simple training script for taxi tip classifier.

Usage:
    python scripts/train_simple.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from taxi_tips_classifier.model import train_taxi_tip_model


def main():
    """Train a simple taxi tip classifier."""
    
    print("Training NYC Taxi Tip Classifier...")
    
    # Training data URL
    train_data_url = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2020-01.parquet"
    
    # Model parameters
    model_params = {
        "n_estimators": 100,
        "max_depth": 10,
        "random_state": 42
    }
    
    # Train model with sample for demo
    classifier = train_taxi_tip_model(
        train_data_path=train_data_url,
        sample_size=50000,  # Use sample for faster training
        model_params=model_params,
        save_path="models/simple_taxi_model.joblib"
    )
    
    print("Training completed!")
    print(f"Model saved to: models/simple_taxi_model.joblib")
    
    # Show summary
    summary = classifier.summary()
    print(f"\nModel Summary:")
    print(f"- Features used: {summary['num_features']}")
    print(f"- Training samples: {summary['training_info']['n_samples']}")
    print(f"- Training time: {summary['training_info']['training_time']:.2f} seconds")


if __name__ == "__main__":
    import os
    os.makedirs("models", exist_ok=True)
    main()
