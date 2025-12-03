"""
Feature Extractor for Botnet Detection
Extracts relevant features from network traffic data

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


class FeatureExtractor:
    """Extracts features from network traffic data for botnet detection"""
    
    def __init__(self):
        """Initialize FeatureExtractor"""
        pass
    
    def extract_features(self, df):
        """
        Extract features from network traffic data
        
        Args:
            df: DataFrame containing network traffic data
            
        Returns:
            DataFrame with extracted features
        """
        features_df = df.copy()
        
        # Network traffic features
        if 'packet_count' in df.columns and 'duration' in df.columns:
            features_df['packets_per_second'] = df['packet_count'] / (df['duration'] + 1)
            features_df['avg_packet_size'] = df['total_bytes'] / (df['packet_count'] + 1)
        
        # DNS query features
        if 'dns_queries' in df.columns:
            features_df['dns_query_rate'] = df['dns_queries'] / (df['duration'] + 1)
        
        # Connection features
        if 'connection_count' in df.columns:
            features_df['connections_per_second'] = df['connection_count'] / (df['duration'] + 1)
        
        # Traffic pattern features
        if 'total_bytes' in df.columns and 'duration' in df.columns:
            features_df['bytes_per_second'] = df['total_bytes'] / (df['duration'] + 1)
        
        # Statistical features
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in ['label', 'is_botnet']:
                # Normalize features
                mean_val = features_df[col].mean()
                std_val = features_df[col].std()
                if std_val > 0:
                    features_df[f'{col}_normalized'] = (features_df[col] - mean_val) / std_val
        
        return features_df
    
    def select_features(self, df, target_column='is_botnet'):
        """
        Select relevant features for model training
        
        Args:
            df: DataFrame with extracted features
            target_column: Name of the target column
            
        Returns:
            Tuple of (X, y) for model training
        """
        # Exclude target and non-feature columns
        exclude_cols = [target_column, 'label', 'id', 'timestamp']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols].select_dtypes(include=[np.number])
        y = df[target_column] if target_column in df.columns else None
        
        return X, y
    
    def get_feature_importance(self, model, feature_names):
        """
        Get feature importance from trained model
        
        Args:
            model: Trained model with feature_importances_ attribute
            feature_names: List of feature names
            
        Returns:
            DataFrame with feature importance
        """
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            return importance_df
        else:
            print("Model does not support feature importance")
            return None

