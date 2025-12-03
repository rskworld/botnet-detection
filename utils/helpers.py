"""
Helper Utilities for Botnet Detection Project

Project: Botnet Detection with Machine Learning
Category: ML Projects
Developer: RSK World
Founder: Molla Samser
Designer & Tester: Rima Khatun
Contact: help@rskworld.in, support@rskworld.in
Phone: +91 93305 39277
Address: Nutanhat, Mongolkote, Purba Burdwan, West Bengal, India, 713147
Website: https://rskworld.in
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def plot_confusion_matrix(y_true, y_pred, title='Confusion Matrix', save_path=None):
    """
    Plot confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        title: Plot title
        save_path: Path to save the plot
    """
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Botnet'],
                yticklabels=['Normal', 'Botnet'])
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def plot_feature_distribution(df, feature_col, target_col='is_botnet', save_path=None):
    """
    Plot distribution of a feature by target class
    
    Args:
        df: DataFrame
        feature_col: Feature column name
        target_col: Target column name
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    normal_data = df[df[target_col] == 0][feature_col]
    botnet_data = df[df[target_col] == 1][feature_col]
    
    plt.hist(normal_data, alpha=0.5, label='Normal', bins=30)
    plt.hist(botnet_data, alpha=0.5, label='Botnet', bins=30)
    plt.xlabel(feature_col)
    plt.ylabel('Frequency')
    plt.title(f'Distribution of {feature_col} by Traffic Type')
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def print_model_metrics(y_true, y_pred, model_name='Model'):
    """
    Print model performance metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name: Name of the model
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, 
        f1_score, classification_report
    )
    
    print(f"\n{model_name} Performance Metrics:")
    print("=" * 50)
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred, average='weighted', zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred, average='weighted', zero_division=0):.4f}")
    print(f"F1-Score:  {f1_score(y_true, y_pred, average='weighted', zero_division=0):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))


def create_directories():
    """Create necessary directories for the project"""
    directories = [
        'data/raw',
        'data/processed',
        'data/models',
        'notebooks',
        'scripts',
        'utils',
        'results'
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {dir_path}")


if __name__ == "__main__":
    # Create project directories
    create_directories()

