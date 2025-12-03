"""
Configuration Manager for Botnet Detection Project
Manages project configuration from JSON file

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

import json
from pathlib import Path
from typing import Dict, Any


class ConfigManager:
    """Manages project configuration"""
    
    def __init__(self, config_path='config/config.json'):
        """
        Initialize configuration manager
        
        Args:
            config_path: Path to configuration JSON file
        """
        self.config_path = Path(config_path)
        self.config = self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                return config
            else:
                print(f"Config file not found: {self.config_path}")
                return self.get_default_config()
        except Exception as e:
            print(f"Error loading config: {e}")
            return self.get_default_config()
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "data": {
                "raw_path": "data/raw/",
                "processed_path": "data/processed/",
                "training_data": "data/processed/training_data.csv"
            },
            "models": {
                "save_path": "data/models/",
                "default_model": "random_forest"
            },
            "training": {
                "test_size": 0.2,
                "random_state": 42,
                "cv_folds": 5
            }
        }
    
    def get(self, key_path: str, default=None):
        """
        Get configuration value by dot-separated path
        
        Args:
            key_path: Dot-separated path (e.g., 'data.training_data')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key_path: str, value: Any):
        """
        Set configuration value by dot-separated path
        
        Args:
            key_path: Dot-separated path (e.g., 'data.training_data')
            value: Value to set
        """
        keys = key_path.split('.')
        config = self.config
        
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        config[keys[-1]] = value
    
    def save_config(self):
        """Save configuration to file"""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        print(f"Configuration saved to {self.config_path}")


# Global config instance
_config_instance = None

def get_config(config_path='config/config.json') -> ConfigManager:
    """Get global configuration instance"""
    global _config_instance
    if _config_instance is None:
        _config_instance = ConfigManager(config_path)
    return _config_instance

