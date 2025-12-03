"""
Visualization Dashboard for Botnet Detection
Creates comprehensive visualizations for data analysis and model performance

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
import sys
sys.path.append(str(Path(__file__).parent))
from feature_extractor import FeatureExtractor
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class VisualizationDashboard:
    """Creates visualizations for botnet detection analysis"""
    
    def __init__(self, data_path, output_dir='results/visualizations/'):
        """
        Initialize dashboard
        
        Args:
            data_path: Path to data file
            output_dir: Directory to save visualizations
        """
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.data = None
        self.feature_extractor = FeatureExtractor()
    
    def load_data(self):
        """Load data"""
        try:
            self.data = pd.read_csv(self.data_path)
            print(f"Data loaded: {len(self.data)} records")
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def plot_data_overview(self):
        """Create overview visualizations of the dataset"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Class distribution
        if 'is_botnet' in self.data.columns:
            ax = axes[0, 0]
            class_counts = self.data['is_botnet'].value_counts()
            colors = ['#2ecc71', '#e74c3c']
            ax.pie(class_counts.values, labels=['Normal', 'Botnet'], autopct='%1.1f%%',
                  colors=colors, startangle=90)
            ax.set_title('Class Distribution', fontsize=14, fontweight='bold')
        
        # Feature distributions
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            col = numeric_cols[0]
            ax = axes[0, 1]
            if 'is_botnet' in self.data.columns:
                normal_data = self.data[self.data['is_botnet'] == 0][col]
                botnet_data = self.data[self.data['is_botnet'] == 1][col]
                ax.hist(normal_data, alpha=0.6, label='Normal', bins=30, color='green')
                ax.hist(botnet_data, alpha=0.6, label='Botnet', bins=30, color='red')
                ax.legend()
            else:
                ax.hist(self.data[col], bins=30, color='blue')
            ax.set_xlabel(col)
            ax.set_ylabel('Frequency')
            ax.set_title(f'Distribution of {col}', fontsize=14, fontweight='bold')
        
        # Correlation heatmap
        if len(numeric_cols) > 1:
            ax = axes[1, 0]
            corr_data = self.data[numeric_cols[:10]]  # Limit to 10 features
            correlation = corr_data.corr()
            sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', 
                       center=0, ax=ax, square=True)
            ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
        
        # Statistics summary
        ax = axes[1, 1]
        ax.axis('off')
        if len(numeric_cols) > 0:
            stats_text = "Dataset Statistics:\n\n"
            stats_text += f"Total Records: {len(self.data)}\n"
            stats_text += f"Features: {len(numeric_cols)}\n"
            if 'is_botnet' in self.data.columns:
                stats_text += f"Normal Traffic: {(self.data['is_botnet'] == 0).sum()}\n"
                stats_text += f"Botnet Traffic: {(self.data['is_botnet'] == 1).sum()}\n"
            ax.text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center',
                   family='monospace')
        
        plt.suptitle('Botnet Detection Dataset Overview', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        output_path = self.output_dir / 'data_overview.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved overview plot to {output_path}")
        plt.close()
    
    def plot_feature_analysis(self):
        """Create detailed feature analysis plots"""
        # Extract features
        data_features = self.feature_extractor.extract_features(self.data)
        X, y = self.feature_extractor.select_features(data_features)
        
        if y is None or len(X.columns) == 0:
            print("Cannot create feature analysis: missing target or features")
            return
        
        # Select top features for visualization
        n_features = min(6, len(X.columns))
        top_features = X.columns[:n_features]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()
        
        for idx, feature in enumerate(top_features):
            ax = axes[idx]
            normal_data = X[y == 0][feature]
            botnet_data = X[y == 1][feature]
            
            ax.hist(normal_data, alpha=0.6, label='Normal', bins=30, color='#2ecc71')
            ax.hist(botnet_data, alpha=0.6, label='Botnet', bins=30, color='#e74c3c')
            ax.set_xlabel(feature, fontsize=10)
            ax.set_ylabel('Frequency', fontsize=10)
            ax.set_title(f'{feature}', fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Feature Distribution Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        output_path = self.output_dir / 'feature_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved feature analysis to {output_path}")
        plt.close()
    
    def plot_traffic_patterns(self):
        """Visualize network traffic patterns"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        if 'is_botnet' not in self.data.columns:
            print("Cannot plot traffic patterns: missing is_botnet column")
            return
        
        # Packet count vs duration
        ax = axes[0, 0]
        normal = self.data[self.data['is_botnet'] == 0]
        botnet = self.data[self.data['is_botnet'] == 1]
        
        if 'packet_count' in self.data.columns and 'duration' in self.data.columns:
            ax.scatter(normal['duration'], normal['packet_count'], 
                      alpha=0.5, label='Normal', color='green', s=20)
            ax.scatter(botnet['duration'], botnet['packet_count'], 
                      alpha=0.5, label='Botnet', color='red', s=20)
            ax.set_xlabel('Duration (seconds)')
            ax.set_ylabel('Packet Count')
            ax.set_title('Packet Count vs Duration', fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Bytes per second
        ax = axes[0, 1]
        if 'total_bytes' in self.data.columns and 'duration' in self.data.columns:
            normal_bps = normal['total_bytes'] / (normal['duration'] + 1)
            botnet_bps = botnet['total_bytes'] / (botnet['duration'] + 1)
            ax.boxplot([normal_bps, botnet_bps], labels=['Normal', 'Botnet'])
            ax.set_ylabel('Bytes per Second')
            ax.set_title('Traffic Rate Comparison', fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        # Connection count
        ax = axes[1, 0]
        if 'connection_count' in self.data.columns:
            ax.hist(normal['connection_count'], alpha=0.6, label='Normal', 
                   bins=30, color='green')
            ax.hist(botnet['connection_count'], alpha=0.6, label='Botnet', 
                   bins=30, color='red')
            ax.set_xlabel('Connection Count')
            ax.set_ylabel('Frequency')
            ax.set_title('Connection Count Distribution', fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # DNS queries
        ax = axes[1, 1]
        if 'dns_queries' in self.data.columns:
            ax.hist(normal['dns_queries'], alpha=0.6, label='Normal', 
                   bins=30, color='green')
            ax.hist(botnet['dns_queries'], alpha=0.6, label='Botnet', 
                   bins=30, color='red')
            ax.set_xlabel('DNS Queries')
            ax.set_ylabel('Frequency')
            ax.set_title('DNS Query Distribution', fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Network Traffic Pattern Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        output_path = self.output_dir / 'traffic_patterns.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved traffic patterns to {output_path}")
        plt.close()
    
    def generate_all_visualizations(self):
        """Generate all visualizations"""
        if not self.load_data():
            return
        
        print("\nGenerating visualizations...")
        self.plot_data_overview()
        self.plot_feature_analysis()
        self.plot_traffic_patterns()
        print("\nAll visualizations generated successfully!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate visualization dashboard')
    parser.add_argument('--data', '-d', default='data/processed/training_data.csv',
                       help='Path to data file')
    parser.add_argument('--output', '-o', default='results/visualizations/',
                       help='Output directory for visualizations')
    
    args = parser.parse_args()
    
    dashboard = VisualizationDashboard(args.data, args.output)
    dashboard.generate_all_visualizations()

