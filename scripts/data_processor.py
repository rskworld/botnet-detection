"""
Data Processor for Botnet Detection
Processes raw network traffic data and prepares it for machine learning

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


class DataProcessor:
    """Processes network traffic data for botnet detection"""
    
    def __init__(self, data_path):
        """
        Initialize DataProcessor
        
        Args:
            data_path: Path to the data file
        """
        self.data_path = data_path
        self.data = None
    
    def load_data(self):
        """Load data from CSV file"""
        try:
            self.data = pd.read_csv(self.data_path)
            print(f"Data loaded successfully: {len(self.data)} records")
            return self.data
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def clean_data(self):
        """Clean and preprocess the data"""
        if self.data is None:
            print("No data loaded. Please load data first.")
            return None
        
        # Remove duplicates
        initial_count = len(self.data)
        self.data = self.data.drop_duplicates()
        print(f"Removed {initial_count - len(self.data)} duplicate records")
        
        # Handle missing values
        missing_count = self.data.isnull().sum().sum()
        if missing_count > 0:
            # Fill numeric columns with median
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            self.data[numeric_cols] = self.data[numeric_cols].fillna(
                self.data[numeric_cols].median()
            )
            print(f"Handled {missing_count} missing values")
        
        return self.data
    
    def get_info(self):
        """Get information about the dataset"""
        if self.data is None:
            print("No data loaded.")
            return
        
        print("\nDataset Information:")
        print(f"Shape: {self.data.shape}")
        print(f"\nColumns: {list(self.data.columns)}")
        print(f"\nData Types:\n{self.data.dtypes}")
        print(f"\nMissing Values:\n{self.data.isnull().sum()}")
        print(f"\nFirst few rows:\n{self.data.head()}")


if __name__ == "__main__":
    # Example usage
    processor = DataProcessor("data/raw/network_traffic.csv")
    data = processor.load_data()
    if data is not None:
        processor.clean_data()
        processor.get_info()

