"""
Quick Start Script for Botnet Detection Project
Generates sample data and trains initial model

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

import sys
from pathlib import Path

# Add scripts to path
sys.path.append(str(Path(__file__).parent / 'scripts'))

from generate_sample_data import generate_sample_data
from train_model import BotnetDetectionTrainer

def main():
    print("=" * 60)
    print("Botnet Detection - Quick Start")
    print("=" * 60)
    
    # Step 1: Generate sample data
    print("\n[1/2] Generating sample data...")
    try:
        generate_sample_data(n_samples=1000, botnet_ratio=0.3)
        print("✓ Sample data generated successfully")
    except Exception as e:
        print(f"✗ Error generating data: {e}")
        return
    
    # Step 2: Train model
    print("\n[2/2] Training model...")
    try:
        trainer = BotnetDetectionTrainer('data/processed/training_data.csv')
        trainer.run()
        print("✓ Model training completed")
    except Exception as e:
        print(f"✗ Error training model: {e}")
        return
    
    print("\n" + "=" * 60)
    print("Quick start completed successfully!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Run detection: python scripts/detect_botnet.py --input data/processed/training_data.csv")
    print("2. Open Jupyter notebook: jupyter notebook notebooks/botnet_detection_analysis.ipynb")
    print("=" * 60)

if __name__ == "__main__":
    main()

