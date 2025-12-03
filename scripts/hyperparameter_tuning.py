"""
Hyperparameter Tuning for Botnet Detection Models
Optimizes model hyperparameters using GridSearchCV and RandomizedSearchCV

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
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import make_scorer, f1_score, accuracy_score
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from feature_extractor import FeatureExtractor
import warnings
warnings.filterwarnings('ignore')


class HyperparameterTuner:
    """Tunes hyperparameters for botnet detection models"""
    
    def __init__(self, data_path, model_save_path='data/models/'):
        """
        Initialize tuner
        
        Args:
            data_path: Path to training data
            model_save_path: Path to save tuned models
        """
        self.data_path = data_path
        self.model_save_path = Path(model_save_path)
        self.model_save_path.mkdir(parents=True, exist_ok=True)
        self.feature_extractor = FeatureExtractor()
        self.best_models = {}
        self.tuning_results = {}
    
    def load_and_prepare_data(self):
        """Load and prepare data for tuning"""
        try:
            data = pd.read_csv(self.data_path)
            print(f"Data loaded: {len(data)} records")
            
            # Extract features
            data = self.feature_extractor.extract_features(data)
            X, y = self.feature_extractor.select_features(data)
            
            if y is None:
                y = (X.iloc[:, 0] > X.iloc[:, 0].median()).astype(int)
            
            # Remove NaN
            mask = ~(X.isnull().any(axis=1) | y.isnull())
            X = X[mask]
            y = y[mask]
            
            self.X = X
            self.y = y
            self.feature_names = list(X.columns)
            
            print(f"Prepared data: {len(X)} samples, {len(self.feature_names)} features")
            return True
        except Exception as e:
            print(f"Error preparing data: {e}")
            return False
    
    def tune_random_forest(self, method='randomized', n_iter=50, cv=5):
        """Tune Random Forest hyperparameters"""
        print("\n" + "="*60)
        print("Tuning Random Forest Classifier")
        print("="*60)
        
        # Define parameter grid
        param_grid = {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [5, 10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }
        
        # Create base model
        base_model = RandomForestClassifier(random_state=42, n_jobs=-1)
        
        # Scoring metric
        scorer = make_scorer(f1_score, average='weighted')
        
        # Perform tuning
        if method == 'grid':
            search = GridSearchCV(
                base_model, param_grid, 
                scoring=scorer, cv=cv, 
                n_jobs=-1, verbose=1
            )
        else:
            search = RandomizedSearchCV(
                base_model, param_grid,
                n_iter=n_iter, scoring=scorer, cv=cv,
                n_jobs=-1, verbose=1, random_state=42
            )
        
        search.fit(self.X, self.y)
        
        # Store results
        self.best_models['random_forest'] = search.best_estimator_
        self.tuning_results['random_forest'] = {
            'best_params': search.best_params_,
            'best_score': search.best_score_,
            'best_cv_score': search.best_score_
        }
        
        print(f"\nBest Parameters: {search.best_params_}")
        print(f"Best CV Score: {search.best_score_:.4f}")
        
        return search.best_estimator_
    
    def tune_gradient_boosting(self, method='randomized', n_iter=50, cv=5):
        """Tune Gradient Boosting hyperparameters"""
        print("\n" + "="*60)
        print("Tuning Gradient Boosting Classifier")
        print("="*60)
        
        # Define parameter grid
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2, 0.3],
            'max_depth': [3, 5, 7, 10],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'subsample': [0.8, 0.9, 1.0]
        }
        
        # Create base model
        base_model = GradientBoostingClassifier(random_state=42)
        
        # Scoring metric
        scorer = make_scorer(f1_score, average='weighted')
        
        # Perform tuning
        if method == 'grid':
            search = GridSearchCV(
                base_model, param_grid,
                scoring=scorer, cv=cv,
                n_jobs=-1, verbose=1
            )
        else:
            search = RandomizedSearchCV(
                base_model, param_grid,
                n_iter=n_iter, scoring=scorer, cv=cv,
                n_jobs=-1, verbose=1, random_state=42
            )
        
        search.fit(self.X, self.y)
        
        # Store results
        self.best_models['gradient_boosting'] = search.best_estimator_
        self.tuning_results['gradient_boosting'] = {
            'best_params': search.best_params_,
            'best_score': search.best_score_,
            'best_cv_score': search.best_score_
        }
        
        print(f"\nBest Parameters: {search.best_params_}")
        print(f"Best CV Score: {search.best_score_:.4f}")
        
        return search.best_estimator_
    
    def save_tuned_models(self):
        """Save tuned models and results"""
        for name, model in self.best_models.items():
            model_path = self.model_save_path / f'{name}_tuned_model.pkl'
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            print(f"Saved tuned {name} model to {model_path}")
        
        # Save tuning results
        results_path = self.model_save_path / 'tuning_results.json'
        with open(results_path, 'w') as f:
            json.dump(self.tuning_results, f, indent=2, default=str)
        print(f"Saved tuning results to {results_path}")
        
        # Save feature names
        feature_path = self.model_save_path / 'feature_names.json'
        with open(feature_path, 'w') as f:
            json.dump(self.feature_names, f)
    
    def run(self, models=['random_forest', 'gradient_boosting'], method='randomized', n_iter=50):
        """Run hyperparameter tuning for specified models"""
        if not self.load_and_prepare_data():
            return
        
        for model_name in models:
            if model_name == 'random_forest':
                self.tune_random_forest(method=method, n_iter=n_iter)
            elif model_name == 'gradient_boosting':
                self.tune_gradient_boosting(method=method, n_iter=n_iter)
        
        self.save_tuned_models()
        print("\n" + "="*60)
        print("Hyperparameter tuning completed!")
        print("="*60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Tune hyperparameters for botnet detection models')
    parser.add_argument('--data', '-d', default='data/processed/training_data.csv', 
                       help='Path to training data')
    parser.add_argument('--models', '-m', nargs='+', 
                       default=['random_forest', 'gradient_boosting'],
                       help='Models to tune')
    parser.add_argument('--method', choices=['grid', 'randomized'], 
                       default='randomized', help='Tuning method')
    parser.add_argument('--n-iter', type=int, default=50, 
                       help='Number of iterations for randomized search')
    
    args = parser.parse_args()
    
    tuner = HyperparameterTuner(args.data)
    tuner.run(models=args.models, method=args.method, n_iter=args.n_iter)

