"""
Report Generator for Botnet Detection Project
Generates comprehensive reports with analysis results

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
from pathlib import Path
from datetime import datetime
import json
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from config_manager import get_config


class ReportGenerator:
    """Generates comprehensive analysis reports"""
    
    def __init__(self, output_dir='results/reports/'):
        """
        Initialize report generator
        
        Args:
            output_dir: Directory to save reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = get_config()
    
    def generate_training_report(self, training_results, model_info=None):
        """
        Generate training report
        
        Args:
            training_results: Dictionary with training results
            model_info: Additional model information
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.output_dir / f'training_report_{timestamp}.txt'
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("BOTNET DETECTION - TRAINING REPORT\n")
            f.write("="*80 + "\n\n")
            
            # Project info
            project_info = self.config.get('project', {})
            f.write("Project Information:\n")
            f.write(f"  Name: {project_info.get('name', 'Botnet Detection')}\n")
            f.write(f"  Version: {project_info.get('version', '1.0.0')}\n")
            f.write(f"  Developer: {project_info.get('developer', 'RSK World')}\n")
            f.write(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Training results
            f.write("="*80 + "\n")
            f.write("TRAINING RESULTS\n")
            f.write("="*80 + "\n\n")
            
            if isinstance(training_results, dict):
                for model_name, results in training_results.items():
                    f.write(f"\nModel: {model_name.upper()}\n")
                    f.write("-" * 80 + "\n")
                    
                    if isinstance(results, dict):
                        for metric, value in results.items():
                            if metric != 'confusion_matrix':
                                if isinstance(value, float):
                                    f.write(f"  {metric.replace('_', ' ').title()}: {value:.4f}\n")
                                else:
                                    f.write(f"  {metric.replace('_', ' ').title()}: {value}\n")
                            else:
                                f.write(f"\n  Confusion Matrix:\n")
                                cm = np.array(value)
                                f.write(f"    Normal  Botnet\n")
                                f.write(f"  Normal  {cm[0,0]:5d}  {cm[0,1]:5d}\n")
                                f.write(f"  Botnet  {cm[1,0]:5d}  {cm[1,1]:5d}\n")
            
            # Model info
            if model_info:
                f.write("\n" + "="*80 + "\n")
                f.write("MODEL INFORMATION\n")
                f.write("="*80 + "\n\n")
                f.write(json.dumps(model_info, indent=2))
            
            f.write("\n\n" + "="*80 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*80 + "\n")
        
        print(f"Training report saved to {report_path}")
        return report_path
    
    def generate_detection_report(self, detection_results, input_file=None):
        """
        Generate detection report
        
        Args:
            detection_results: DataFrame with detection results
            input_file: Path to input file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.output_dir / f'detection_report_{timestamp}.txt'
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("BOTNET DETECTION - DETECTION REPORT\n")
            f.write("="*80 + "\n\n")
            
            # Report info
            f.write("Detection Information:\n")
            f.write(f"  Input File: {input_file or 'N/A'}\n")
            f.write(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"  Total Records: {len(detection_results)}\n\n")
            
            # Detection summary
            if 'is_botnet' in detection_results.columns:
                botnet_count = detection_results['is_botnet'].sum()
                normal_count = len(detection_results) - botnet_count
                
                f.write("="*80 + "\n")
                f.write("DETECTION SUMMARY\n")
                f.write("="*80 + "\n\n")
                f.write(f"Normal Traffic:  {normal_count:6d} ({normal_count/len(detection_results)*100:5.2f}%)\n")
                f.write(f"Botnet Traffic:  {botnet_count:6d} ({botnet_count/len(detection_results)*100:5.2f}%)\n")
                f.write(f"Total Records:   {len(detection_results):6d} (100.00%)\n\n")
                
                # Probability distribution
                if 'botnet_probability' in detection_results.columns:
                    f.write("Probability Distribution:\n")
                    f.write(f"  Mean:   {detection_results['botnet_probability'].mean():.4f}\n")
                    f.write(f"  Median: {detection_results['botnet_probability'].median():.4f}\n")
                    f.write(f"  Min:    {detection_results['botnet_probability'].min():.4f}\n")
                    f.write(f"  Max:    {detection_results['botnet_probability'].max():.4f}\n\n")
                
                # Top botnet detections
                if 'botnet_probability' in detection_results.columns:
                    f.write("Top 10 Botnet Detections (by probability):\n")
                    f.write("-" * 80 + "\n")
                    top_botnets = detection_results.nlargest(10, 'botnet_probability')
                    for idx, row in top_botnets.iterrows():
                        f.write(f"  Record {idx}: Probability = {row['botnet_probability']:.4f}\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*80 + "\n")
        
        print(f"Detection report saved to {report_path}")
        return report_path
    
    def generate_summary_report(self, data_stats=None, model_stats=None, detection_stats=None):
        """
        Generate comprehensive summary report
        
        Args:
            data_stats: Data statistics
            model_stats: Model statistics
            detection_stats: Detection statistics
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.output_dir / f'summary_report_{timestamp}.html'
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Botnet Detection Summary Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        h2 {{ color: #666; border-bottom: 2px solid #666; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        .stats {{ background-color: #f2f2f2; padding: 15px; margin: 10px 0; }}
    </style>
</head>
<body>
    <h1>Botnet Detection - Summary Report</h1>
    <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    
    <h2>Project Information</h2>
    <div class="stats">
        <p><strong>Project:</strong> Botnet Detection with Machine Learning</p>
        <p><strong>Developer:</strong> RSK World</p>
        <p><strong>Founder:</strong> Molla Samser</p>
        <p><strong>Designer & Tester:</strong> Rima Khatun</p>
    </div>
"""
        
        if data_stats:
            html_content += """
    <h2>Data Statistics</h2>
    <div class="stats">
"""
            for key, value in data_stats.items():
                html_content += f"        <p><strong>{key}:</strong> {value}</p>\n"
            html_content += "    </div>\n"
        
        if model_stats:
            html_content += """
    <h2>Model Statistics</h2>
    <div class="stats">
"""
            for key, value in model_stats.items():
                html_content += f"        <p><strong>{key}:</strong> {value}</p>\n"
            html_content += "    </div>\n"
        
        if detection_stats:
            html_content += """
    <h2>Detection Statistics</h2>
    <div class="stats">
"""
            for key, value in detection_stats.items():
                html_content += f"        <p><strong>{key}:</strong> {value}</p>\n"
            html_content += "    </div>\n"
        
        html_content += """
</body>
</html>
"""
        
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        print(f"Summary report saved to {report_path}")
        return report_path

