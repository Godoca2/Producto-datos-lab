"""
Visualization utilities for taxi tip classification analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns


def plot_confusion_matrix(confusion_matrix: np.ndarray,
                         class_names: List[str] = None,
                         title: str = "Confusion Matrix",
                         figsize: Tuple[int, int] = (8, 6)) -> plt.Figure:
    """
    Plot confusion matrix as a heatmap.
    
    Args:
        confusion_matrix: 2D array with confusion matrix values
        class_names: Names for the classes (default: ["Low Tip", "High Tip"])
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure object
    """
    if class_names is None:
        class_names = ["Low Tip", "High Tip"]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Convert to numpy array if needed
    if isinstance(confusion_matrix, list):
        confusion_matrix = np.array(confusion_matrix)
    
    # Create heatmap
    sns.heatmap(confusion_matrix, 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                ax=ax)
    
    ax.set_title(title)
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    
    plt.tight_layout()
    return fig


def plot_feature_importance(model,
                           feature_names: List[str],
                           title: str = "Feature Importance",
                           top_n: int = 15,
                           figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
    """
    Plot feature importance from a trained model.
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: List of feature names
        title: Plot title
        top_n: Number of top features to show
        figsize: Figure size
        
    Returns:
        Matplotlib figure object
    """
    # Get feature importances
    importances = model.feature_importances_
    
    # Create DataFrame for easier manipulation
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    # Select top N features
    top_features = feature_importance_df.head(top_n)
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    bars = ax.barh(range(len(top_features)), top_features['importance'].values)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'].values)
    ax.set_xlabel('Importance')
    ax.set_title(title)
    
    # Invert y-axis to have most important at top
    ax.invert_yaxis()
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                f'{width:.3f}', ha='left', va='center')
    
    plt.tight_layout()
    return fig


def plot_prediction_distribution(predictions: Dict[str, Any],
                                title: str = "Prediction Probability Distribution",
                                figsize: Tuple[int, int] = (12, 5)) -> plt.Figure:
    """
    Plot distribution of prediction probabilities.
    
    Args:
        predictions: Dictionary with prediction results
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure object
    """
    probabilities = np.array(predictions['predictions']['probabilities'])
    true_labels = np.array(predictions['predictions']['true_labels'])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot 1: Overall distribution
    ax1.hist(probabilities, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_xlabel('Prediction Probability')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Overall Probability Distribution')
    ax1.axvline(x=0.5, color='red', linestyle='--', label='Decision Threshold')
    ax1.legend()
    
    # Plot 2: Distribution by true class
    high_tip_probs = probabilities[true_labels == 1]
    low_tip_probs = probabilities[true_labels == 0]
    
    ax2.hist(low_tip_probs, bins=30, alpha=0.7, label='True Low Tip', color='orange')
    ax2.hist(high_tip_probs, bins=30, alpha=0.7, label='True High Tip', color='green')
    ax2.set_xlabel('Prediction Probability')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Probability Distribution by True Class')
    ax2.axvline(x=0.5, color='red', linestyle='--', label='Decision Threshold')
    ax2.legend()
    
    plt.suptitle(title)
    plt.tight_layout()
    return fig


def plot_model_comparison(comparison_results: Dict[str, Dict[str, Any]],
                         metric: str = "f1_score",
                         title: str = "Model Performance Comparison",
                         figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """
    Plot comparison of model performance across different datasets.
    
    Args:
        comparison_results: Dictionary with results for different datasets
        metric: Metric to plot (e.g., 'f1_score', 'accuracy', 'precision', 'recall')
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure object
    """
    datasets = list(comparison_results.keys())
    metric_values = [comparison_results[dataset][metric] for dataset in datasets]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    bars = ax.bar(datasets, metric_values, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
    
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(title)
    ax.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, value in zip(bars, metric_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig


def plot_class_distribution(data: pd.DataFrame,
                           target_col: str = "high_tip",
                           dataset_name: str = "Dataset",
                           figsize: Tuple[int, int] = (8, 6)) -> plt.Figure:
    """
    Plot distribution of target classes.
    
    Args:
        data: DataFrame with data
        target_col: Name of target column
        dataset_name: Name of the dataset for title
        figsize: Figure size
        
    Returns:
        Matplotlib figure object
    """
    class_counts = data[target_col].value_counts()
    class_labels = ['Low Tip', 'High Tip']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Bar plot
    bars = ax1.bar(class_labels, class_counts.values, color=['orange', 'green'])
    ax1.set_ylabel('Count')
    ax1.set_title(f'{dataset_name} - Class Distribution')
    
    # Add value labels
    for bar, count in zip(bars, class_counts.values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 50,
                f'{count}', ha='center', va='bottom')
    
    # Pie chart
    ax2.pie(class_counts.values, labels=class_labels, autopct='%1.1f%%', 
            colors=['orange', 'green'], startangle=90)
    ax2.set_title(f'{dataset_name} - Class Proportion')
    
    plt.tight_layout()
    return fig


def save_plot(fig: plt.Figure, filepath: str, dpi: int = 300) -> None:
    """
    Save a matplotlib figure to file.
    
    Args:
        fig: Matplotlib figure to save
        filepath: Path where to save the figure
        dpi: Resolution for saved figure
    """
    try:
        fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
        print(f"Plot saved to {filepath}")
    except Exception as e:
        print(f"Error saving plot: {e}")
        raise


def create_evaluation_report(model,
                           test_data: pd.DataFrame,
                           features: List[str],
                           target_col: str = "high_tip",
                           save_dir: Optional[str] = None) -> Dict[str, plt.Figure]:
    """
    Create a comprehensive evaluation report with multiple plots.
    
    Args:
        model: Trained model
        test_data: Test dataset
        features: List of feature names
        target_col: Name of target column
        save_dir: Optional directory to save plots
        
    Returns:
        Dictionary of plot names and figure objects
    """
    from .predict import predict_and_evaluate
    
    # Get predictions and evaluation results
    results = predict_and_evaluate(model, test_data, features, target_col)
    
    plots = {}
    
    # 1. Confusion Matrix
    plots['confusion_matrix'] = plot_confusion_matrix(
        results['confusion_matrix'],
        title="Confusion Matrix - Test Data"
    )
    
    # 2. Feature Importance
    plots['feature_importance'] = plot_feature_importance(
        model, features,
        title="Feature Importance"
    )
    
    # 3. Prediction Distribution
    plots['prediction_distribution'] = plot_prediction_distribution(
        results,
        title="Prediction Probability Distribution"
    )
    
    # 4. Class Distribution
    plots['class_distribution'] = plot_class_distribution(
        test_data, target_col,
        dataset_name="Test Data"
    )
    
    # Save plots if directory provided
    if save_dir is not None:
        import os
        os.makedirs(save_dir, exist_ok=True)
        for plot_name, fig in plots.items():
            filepath = os.path.join(save_dir, f"{plot_name}.png")
            save_plot(fig, filepath)
    
    return plots
