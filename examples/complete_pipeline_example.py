#!/usr/bin/env python3
"""
Example script demonstrating the complete taxi tip classification pipeline.

This script shows how to use the modular components to:
1. Load and preprocess data
2. Train a model
3. Evaluate on test data
4. Make predictions
5. Generate visualizations

Usage:
    python examples/complete_pipeline_example.py
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from taxi_tips_classifier.model import TaxiTipClassifier
from taxi_tips_classifier.features import get_feature_info


def main():
    """Run the complete pipeline example."""
    
    print("=" * 60)
    print("NYC Taxi Tip Classifier - Complete Pipeline Example")
    print("=" * 60)
    
    # URLs for NYC taxi data
    train_data_url = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2020-01.parquet"
    test_data_url = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2020-02.parquet"
    may_data_url = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2020-05.parquet"
    
    # Model save path
    model_save_path = "models/taxi_tip_classifier.joblib"
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    print("\n1. Creating and configuring classifier...")
    
    # Create classifier with custom parameters
    model_params = {
        "n_estimators": 100,
        "max_depth": 10,
        "random_state": 42,
        "n_jobs": -1  # Use all available cores
    }
    
    classifier = TaxiTipClassifier(model_params=model_params)
    
    # Print feature information
    feature_info = get_feature_info()
    print(f"Total features: {feature_info['total_features']}")
    print(f"Numeric features: {len(feature_info['numeric_features'])}")
    print(f"Categorical features: {len(feature_info['categorical_features'])}")
    
    print("\n2. Training model on January 2020 data...")
    
    # Train the model (using a sample for faster execution)
    sample_size = 100000  # Use smaller sample for demo
    training_info = classifier.train(
        training_data=train_data_url,
        sample_size=sample_size
    )
    
    print(f"Training completed in {training_info['training_time']:.2f} seconds")
    print(f"Trained on {training_info['n_samples']} samples")
    
    print("\n3. Evaluating on February 2020 data...")
    
    # Evaluate on February data
    feb_results = classifier.evaluate(test_data_url)
    print(f"February F1-Score: {feb_results['f1_score']:.4f}")
    print(f"February Accuracy: {feb_results['accuracy']:.4f}")
    
    print("\n4. Evaluating on May 2020 data...")
    
    # Evaluate on May data to see performance degradation
    may_results = classifier.evaluate(may_data_url)
    print(f"May F1-Score: {may_results['f1_score']:.4f}")
    print(f"May Accuracy: {may_results['accuracy']:.4f}")
    
    print("\n5. Saving trained model...")
    
    # Save the model
    classifier.save_model(model_save_path)
    
    print("\n6. Feature importance analysis...")
    
    # Get and display feature importance
    feature_importance = classifier.get_feature_importance()
    print("Top 10 most important features:")
    print(feature_importance.head(10).to_string(index=False))
    
    print("\n7. Making single prediction example...")
    
    # Load a small sample for single prediction demo
    test_df = classifier.load_and_preprocess_data(test_data_url)
    sample_trip = test_df.iloc[0]
    
    # Make prediction for single trip
    prob = classifier.predict(sample_trip, return_proba=True)
    binary_pred = classifier.predict(sample_trip, return_proba=False)
    actual = sample_trip[classifier.target_col]
    
    print(f"Sample trip prediction:")
    print(f"  Probability of high tip: {prob:.4f}")
    print(f"  Binary prediction: {'High Tip' if binary_pred else 'Low Tip'}")
    print(f"  Actual label: {'High Tip' if actual else 'Low Tip'}")
    
    print("\n8. Performance comparison summary...")
    
    # Compare performance across datasets
    comparison_data = {
        "February 2020": feb_results,
        "May 2020": may_results
    }
    
    print(f"{'Dataset':<15} {'F1-Score':<10} {'Accuracy':<10} {'Precision':<10} {'Recall':<10}")
    print("-" * 65)
    for dataset_name, results in comparison_data.items():
        f1 = results['f1_score']
        acc = results['accuracy']
        prec = results['precision']
        rec = results['recall']
        print(f"{dataset_name:<15} {f1:<10.4f} {acc:<10.4f} {prec:<10.4f} {rec:<10.4f}")
    
    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")
    print(f"Model saved to: {model_save_path}")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExecution interrupted by user.")
    except Exception as e:
        print(f"\nError during execution: {e}")
        raise
