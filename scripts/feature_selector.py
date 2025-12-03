"""
Feature Selection Utilities for Botnet Detection
Implements various feature selection methods

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
from sklearn.feature_selection import (
    SelectKBest, f_classif, mutual_info_classif,
    RFE, SelectFromModel, chi2
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from feature_extractor import FeatureExtractor


class FeatureSelector:
    """Implements various feature selection methods"""
    
    def __init__(self):
        """Initialize feature selector"""
        self.selected_features = None
        self.feature_scores = None
    
    def select_k_best(self, X, y, k=10, score_func=f_classif):
        """
        Select K best features using statistical tests
        
        Args:
            X: Feature matrix
            y: Target vector
            k: Number of features to select
            score_func: Scoring function (f_classif, chi2, mutual_info_classif)
            
        Returns:
            Selected feature names
        """
        # Handle non-negative requirement for chi2
        if score_func == chi2:
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X)
            X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        else:
            X_scaled = X
        
        selector = SelectKBest(score_func=score_func, k=min(k, X.shape[1]))
        selector.fit(X_scaled, y)
        
        selected_indices = selector.get_support(indices=True)
        self.selected_features = X.columns[selected_indices].tolist()
        self.feature_scores = pd.DataFrame({
            'feature': X.columns,
            'score': selector.scores_
        }).sort_values('score', ascending=False)
        
        return self.selected_features
    
    def select_rfe(self, X, y, n_features=10, estimator=None):
        """
        Select features using Recursive Feature Elimination
        
        Args:
            X: Feature matrix
            y: Target vector
            n_features: Number of features to select
            estimator: Base estimator (default: RandomForestClassifier)
            
        Returns:
            Selected feature names
        """
        if estimator is None:
            estimator = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        
        selector = RFE(estimator=estimator, n_features_to_select=min(n_features, X.shape[1]))
        selector.fit(X, y)
        
        self.selected_features = X.columns[selector.get_support()].tolist()
        self.feature_scores = pd.DataFrame({
            'feature': X.columns,
            'rank': selector.ranking_,
            'selected': selector.get_support()
        }).sort_values('rank')
        
        return self.selected_features
    
    def select_from_model(self, X, y, estimator=None, threshold='median'):
        """
        Select features based on model importance
        
        Args:
            X: Feature matrix
            y: Target vector
            estimator: Model with feature_importances_ (default: RandomForestClassifier)
            threshold: Threshold for selection
            
        Returns:
            Selected feature names
        """
        if estimator is None:
            estimator = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        
        estimator.fit(X, y)
        selector = SelectFromModel(estimator, threshold=threshold)
        selector.fit(X, y)
        
        self.selected_features = X.columns[selector.get_support()].tolist()
        
        if hasattr(estimator, 'feature_importances_'):
            self.feature_scores = pd.DataFrame({
                'feature': X.columns,
                'importance': estimator.feature_importances_,
                'selected': selector.get_support()
            }).sort_values('importance', ascending=False)
        
        return self.selected_features
    
    def select_mutual_info(self, X, y, k=10):
        """
        Select features using mutual information
        
        Args:
            X: Feature matrix
            y: Target vector
            k: Number of features to select
            
        Returns:
            Selected feature names
        """
        return self.select_k_best(X, y, k=k, score_func=mutual_info_classif)
    
    def get_feature_importance_ranking(self):
        """Get feature importance ranking"""
        if self.feature_scores is not None:
            return self.feature_scores
        else:
            print("No feature scores available. Run a selection method first.")
            return None
    
    def filter_features(self, X, feature_list=None):
        """
        Filter features from dataset
        
        Args:
            X: Feature matrix
            feature_list: List of features to keep (uses selected_features if None)
            
        Returns:
            Filtered feature matrix
        """
        if feature_list is None:
            feature_list = self.selected_features
        
        if feature_list is None:
            print("No features selected. Run a selection method first.")
            return X
        
        available_features = [f for f in feature_list if f in X.columns]
        return X[available_features]


if __name__ == "__main__":
    # Example usage
    from feature_extractor import FeatureExtractor
    
    # Load sample data
    data = pd.read_csv('data/processed/training_data.csv')
    extractor = FeatureExtractor()
    data_features = extractor.extract_features(data)
    X, y = extractor.select_features(data_features)
    
    if y is None:
        y = (X.iloc[:, 0] > X.iloc[:, 0].median()).astype(int)
    
    # Remove NaN
    mask = ~(X.isnull().any(axis=1) | y.isnull())
    X = X[mask]
    y = y[mask]
    
    # Feature selection
    selector = FeatureSelector()
    
    print("Selecting top 10 features using mutual information...")
    selected = selector.select_mutual_info(X, y, k=10)
    print(f"Selected features: {selected}")
    print(f"\nFeature scores:\n{selector.get_feature_importance_ranking()}")

