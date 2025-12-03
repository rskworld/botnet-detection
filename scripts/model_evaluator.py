"""
Model Evaluator and Comparison Tool
Evaluates and compares multiple models with comprehensive metrics

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
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from feature_extractor import FeatureExtractor
import warnings
warnings.filterwarnings('ignore')


class ModelEvaluator:
    """Evaluates and compares machine learning models"""
    
    def __init__(self, data_path, model_path='data/models/'):
        """
        Initialize evaluator
        
        Args:
            data_path: Path to evaluation data
            model_path: Path to saved models
        """
        self.data_path = data_path
        self.model_path = Path(model_path)
        self.feature_extractor = FeatureExtractor()
        self.results = {}
        self.models = {}
    
    def load_and_prepare_data(self):
        """Load and prepare data for evaluation"""
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
            
            print(f"Prepared data: {len(X)} samples")
            return True
        except Exception as e:
            print(f"Error preparing data: {e}")
            return False
    
    def load_model(self, model_name):
        """Load a trained model"""
        try:
            model_file = self.model_path / f'{model_name}_model.pkl'
            if not model_file.exists():
                model_file = self.model_path / f'{model_name}_tuned_model.pkl'
            
            with open(model_file, 'rb') as f:
                model = pickle.load(f)
            
            self.models[model_name] = model
            return model
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            return None
    
    def evaluate_model(self, model_name, model, X_test=None, y_test=None):
        """Evaluate a single model"""
        if X_test is None:
            X_test = self.X
        if y_test is None:
            y_test = self.y
        
        # Predictions
        y_pred = model.predict(X_test)
        y_proba = None
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        # ROC AUC if probabilities available
        if y_proba is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_test, y_proba)
                metrics['avg_precision'] = average_precision_score(y_test, y_proba)
            except:
                pass
        
        # Cross-validation scores
        cv_scores = cross_val_score(model, self.X, self.y, cv=5, scoring='f1_weighted')
        metrics['cv_mean'] = cv_scores.mean()
        metrics['cv_std'] = cv_scores.std()
        
        self.results[model_name] = {
            'metrics': metrics,
            'predictions': y_pred,
            'probabilities': y_proba
        }
        
        return metrics
    
    def compare_models(self, model_names):
        """Compare multiple models"""
        print("\n" + "="*80)
        print("Model Comparison")
        print("="*80)
        
        # Load and evaluate all models
        for name in model_names:
            model = self.load_model(name)
            if model:
                print(f"\nEvaluating {name}...")
                self.evaluate_model(name, model)
        
        # Create comparison DataFrame
        comparison_data = []
        for name, result in self.results.items():
            metrics = result['metrics']
            comparison_data.append({
                'Model': name,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1_score'],
                'CV Mean': metrics.get('cv_mean', 0),
                'CV Std': metrics.get('cv_std', 0),
                'ROC AUC': metrics.get('roc_auc', 0)
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('F1-Score', ascending=False)
        
        print("\n" + comparison_df.to_string(index=False))
        
        return comparison_df
    
    def plot_comparison(self, save_path=None):
        """Plot model comparison charts"""
        if not self.results:
            print("No results to plot. Run evaluation first.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Metrics comparison
        models = list(self.results.keys())
        metrics_names = ['accuracy', 'precision', 'recall', 'f1_score']
        metrics_values = {
            metric: [self.results[m]['metrics'][metric] for m in models]
            for metric in metrics_names
        }
        
        x = np.arange(len(models))
        width = 0.2
        
        ax = axes[0, 0]
        for i, metric in enumerate(metrics_names):
            ax.bar(x + i*width, metrics_values[metric], width, label=metric.replace('_', ' ').title())
        ax.set_xlabel('Models')
        ax.set_ylabel('Score')
        ax.set_title('Model Performance Metrics')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim([0, 1])
        
        # Confusion matrices
        n_models = len(models)
        for idx, model_name in enumerate(models[:4]):  # Max 4 models
            row = 1 if idx < 2 else 0
            col = idx % 2 if idx < 2 else (idx - 2) % 2
            if idx >= 2:
                row = 1
                col = idx - 2
            
            if row < 2 and col < 2:
                cm = np.array(self.results[model_name]['metrics']['confusion_matrix'])
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[row, col],
                           xticklabels=['Normal', 'Botnet'],
                           yticklabels=['Normal', 'Botnet'])
                axes[row, col].set_title(f'{model_name} Confusion Matrix')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comparison plot saved to {save_path}")
        
        plt.show()
    
    def generate_report(self, output_path='results/model_evaluation_report.txt'):
        """Generate detailed evaluation report"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("Botnet Detection Model Evaluation Report\n")
            f.write("="*80 + "\n\n")
            f.write("Project: Botnet Detection with Machine Learning\n")
            f.write("Developer: RSK World\n")
            f.write("="*80 + "\n\n")
            
            for model_name, result in self.results.items():
                f.write(f"\n{'='*80}\n")
                f.write(f"Model: {model_name}\n")
                f.write(f"{'='*80}\n\n")
                
                metrics = result['metrics']
                f.write("Performance Metrics:\n")
                f.write(f"  Accuracy:  {metrics['accuracy']:.4f}\n")
                f.write(f"  Precision: {metrics['precision']:.4f}\n")
                f.write(f"  Recall:    {metrics['recall']:.4f}\n")
                f.write(f"  F1-Score:  {metrics['f1_score']:.4f}\n")
                if 'roc_auc' in metrics:
                    f.write(f"  ROC AUC:   {metrics['roc_auc']:.4f}\n")
                f.write(f"  CV Mean:   {metrics['cv_mean']:.4f} (+/- {metrics['cv_std']:.4f})\n")
                
                f.write("\nConfusion Matrix:\n")
                cm = np.array(metrics['confusion_matrix'])
                f.write(f"  Normal  Botnet\n")
                f.write(f"Normal  {cm[0,0]:5d}  {cm[0,1]:5d}\n")
                f.write(f"Botnet  {cm[1,0]:5d}  {cm[1,1]:5d}\n")
                f.write("\n")
        
        print(f"Report saved to {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate and compare botnet detection models')
    parser.add_argument('--data', '-d', default='data/processed/training_data.csv',
                       help='Path to evaluation data')
    parser.add_argument('--models', '-m', nargs='+',
                       default=['random_forest', 'gradient_boosting', 'logistic_regression', 'svm'],
                       help='Models to evaluate')
    parser.add_argument('--plot', action='store_true', help='Generate comparison plots')
    parser.add_argument('--report', action='store_true', help='Generate evaluation report')
    
    args = parser.parse_args()
    
    evaluator = ModelEvaluator(args.data)
    if evaluator.load_and_prepare_data():
        comparison_df = evaluator.compare_models(args.models)
        
        if args.plot:
            evaluator.plot_comparison('results/model_comparison.png')
        
        if args.report:
            evaluator.generate_report()

