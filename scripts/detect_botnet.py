"""
Botnet Detection Script
Detects botnet activities in network traffic using trained models

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
import pickle
import json
import argparse
from pathlib import Path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from feature_extractor import FeatureExtractor


class BotnetDetector:
    """Detects botnet activities in network traffic"""
    
    def __init__(self, model_path='data/models/', model_name='random_forest'):
        """
        Initialize detector
        
        Args:
            model_path: Path to saved models
            model_name: Name of the model to use
        """
        self.model_path = Path(model_path)
        self.model_name = model_name
        self.model = None
        self.feature_names = None
        self.feature_extractor = FeatureExtractor()
        self.load_model()
    
    def load_model(self):
        """Load trained model and feature names"""
        try:
            # Load model
            model_file = self.model_path / f'{self.model_name}_model.pkl'
            with open(model_file, 'rb') as f:
                self.model = pickle.load(f)
            
            # Load feature names
            feature_file = self.model_path / 'feature_names.json'
            with open(feature_file, 'r') as f:
                self.feature_names = json.load(f)
            
            print(f"Model loaded: {self.model_name}")
            print(f"Features: {len(self.feature_names)}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def detect(self, data):
        """
        Detect botnet activities in data
        
        Args:
            data: DataFrame with network traffic data
            
        Returns:
            DataFrame with detection results
        """
        if self.model is None:
            print("Model not loaded. Cannot perform detection.")
            return None
        
        # Extract features
        data_processed = self.feature_extractor.extract_features(data)
        
        # Select features
        X, _ = self.feature_extractor.select_features(data_processed)
        
        # Align features with model
        missing_features = set(self.feature_names) - set(X.columns)
        extra_features = set(X.columns) - set(self.feature_names)
        
        if missing_features:
            for feat in missing_features:
                X[feat] = 0
        
        if extra_features:
            X = X.drop(columns=extra_features)
        
        # Reorder columns to match model
        X = X[self.feature_names]
        
        # Make predictions
        predictions = self.model.predict(X)
        probabilities = None
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(X)[:, 1]
        
        # Create results dataframe
        results = data.copy()
        results['is_botnet'] = predictions
        results['botnet_probability'] = probabilities if probabilities is not None else predictions
        
        return results
    
    def detect_from_file(self, input_file, output_file=None):
        """
        Detect botnet activities from a file
        
        Args:
            input_file: Path to input CSV file
            output_file: Path to save results (optional)
            
        Returns:
            DataFrame with detection results
        """
        try:
            data = pd.read_csv(input_file)
            print(f"Loaded {len(data)} records from {input_file}")
            
            results = self.detect(data)
            
            if results is not None:
                botnet_count = results['is_botnet'].sum()
                print(f"\nDetection Results:")
                print(f"Total records: {len(results)}")
                print(f"Botnet detected: {botnet_count} ({botnet_count/len(results)*100:.2f}%)")
                print(f"Normal traffic: {len(results) - botnet_count} ({(len(results) - botnet_count)/len(results)*100:.2f}%)")
                
                if output_file:
                    results.to_csv(output_file, index=False)
                    print(f"\nResults saved to {output_file}")
            
            return results
        except Exception as e:
            print(f"Error processing file: {e}")
            return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Detect botnet activities in network traffic')
    parser.add_argument('--input', '-i', required=True, help='Input CSV file with network traffic data')
    parser.add_argument('--output', '-o', help='Output CSV file for results')
    parser.add_argument('--model', '-m', default='random_forest', help='Model name to use')
    parser.add_argument('--model-path', default='data/models/', help='Path to saved models')
    
    args = parser.parse_args()
    
    detector = BotnetDetector(model_path=args.model_path, model_name=args.model)
    detector.detect_from_file(args.input, args.output)

