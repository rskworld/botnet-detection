"""
Model Training Script for Botnet Detection
Trains machine learning models to detect botnet activities

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
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, classification_report
)
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from feature_extractor import FeatureExtractor
import warnings
warnings.filterwarnings('ignore')


class BotnetDetectionTrainer:
    """Trains machine learning models for botnet detection"""
    
    def __init__(self, data_path, model_save_path='data/models/'):
        """
        Initialize trainer
        
        Args:
            data_path: Path to training data
            model_save_path: Path to save trained models
        """
        self.data_path = data_path
        self.model_save_path = Path(model_save_path)
        self.model_save_path.mkdir(parents=True, exist_ok=True)
        self.feature_extractor = FeatureExtractor()
        self.models = {}
        self.results = {}
    
    def load_data(self):
        """Load training data"""
        try:
            self.data = pd.read_csv(self.data_path)
            print(f"Data loaded: {len(self.data)} records")
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def prepare_data(self):
        """Prepare data for training"""
        # Extract features
        self.data = self.feature_extractor.extract_features(self.data)
        
        # Select features
        X, y = self.feature_extractor.select_features(self.data)
        
        # Handle missing target
        if y is None:
            print("Warning: No target column found. Creating synthetic target for demonstration.")
            # Create synthetic target based on some heuristic
            y = (X.iloc[:, 0] > X.iloc[:, 0].median()).astype(int)
        
        # Remove rows with NaN
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[mask]
        y = y[mask]
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        self.feature_names = list(X.columns)
        print(f"Training set: {len(self.X_train)} samples")
        print(f"Test set: {len(self.X_test)} samples")
        print(f"Features: {len(self.feature_names)}")
        
        return True
    
    def train_models(self):
        """Train multiple ML models"""
        models_config = {
            'random_forest': RandomForestClassifier(
                n_estimators=100, 
                max_depth=10, 
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            ),
            'logistic_regression': LogisticRegression(
                max_iter=1000,
                random_state=42
            ),
            'svm': SVC(
                kernel='rbf',
                probability=True,
                random_state=42
            )
        }
        
        for name, model in models_config.items():
            print(f"\nTraining {name}...")
            try:
                model.fit(self.X_train, self.y_train)
                self.models[name] = model
                
                # Evaluate
                y_pred = model.predict(self.X_test)
                results = {
                    'accuracy': accuracy_score(self.y_test, y_pred),
                    'precision': precision_score(self.y_test, y_pred, average='weighted', zero_division=0),
                    'recall': recall_score(self.y_test, y_pred, average='weighted', zero_division=0),
                    'f1_score': f1_score(self.y_test, y_pred, average='weighted', zero_division=0),
                    'confusion_matrix': confusion_matrix(self.y_test, y_pred).tolist()
                }
                self.results[name] = results
                
                print(f"Accuracy: {results['accuracy']:.4f}")
                print(f"Precision: {results['precision']:.4f}")
                print(f"Recall: {results['recall']:.4f}")
                print(f"F1-Score: {results['f1_score']:.4f}")
                
            except Exception as e:
                print(f"Error training {name}: {e}")
    
    def save_models(self):
        """Save trained models"""
        for name, model in self.models.items():
            model_path = self.model_save_path / f'{name}_model.pkl'
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            print(f"Saved {name} model to {model_path}")
        
        # Save feature names
        with open(self.model_save_path / 'feature_names.json', 'w') as f:
            json.dump(self.feature_names, f)
        
        # Save results
        with open(self.model_save_path / 'training_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Save best model info
        best_model = max(self.results.items(), key=lambda x: x[1]['f1_score'])
        best_model_info = {
            'model_name': best_model[0],
            'metrics': best_model[1]
        }
        with open(self.model_save_path / 'best_model.json', 'w') as f:
            json.dump(best_model_info, f, indent=2)
        
        print(f"\nBest model: {best_model[0]} (F1-Score: {best_model[1]['f1_score']:.4f})")
    
    def run(self):
        """Run complete training pipeline"""
        if not self.load_data():
            return
        
        if not self.prepare_data():
            return
        
        self.train_models()
        self.save_models()
        
        print("\nTraining completed successfully!")


if __name__ == "__main__":
    import sys
    
    # Default data path
    data_path = "data/processed/training_data.csv"
    
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    
    trainer = BotnetDetectionTrainer(data_path)
    trainer.run()

