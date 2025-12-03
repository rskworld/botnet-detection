# Botnet Detection with Machine Learning

<!--
Project: Botnet Detection with Machine Learning
Category: ML Projects
Developer: RSK World
Founder: Molla Samser
Designer & Tester: Rima Khatun
Contact: help@rskworld.in, support@rskworld.in
Phone: +91 93305 39277
Address: Nutanhat, Mongolkote, Purba Burdwan, West Bengal, India, 713147
Website: https://rskworld.in
-->

ML-based system to detect botnet activities and compromised devices in network traffic.

## Description

This project develops a botnet detection system that identifies botnet-infected devices by analyzing network communication patterns, DNS queries, and traffic characteristics. It uses machine learning to distinguish between normal and botnet traffic.

## Features

### Core Features
- Network traffic analysis
- Botnet pattern recognition
- DNS query analysis
- Classification model training
- Detection accuracy metrics

### Advanced Features
- **Hyperparameter Tuning**: Automated optimization using GridSearchCV and RandomizedSearchCV
- **Model Evaluation & Comparison**: Comprehensive evaluation with multiple metrics and cross-validation
- **Visualization Dashboard**: Interactive charts and plots for data analysis
- **Feature Selection**: Multiple feature selection methods (K-Best, RFE, Mutual Information)
- **Configuration Management**: Centralized configuration system
- **Report Generation**: Automated report generation (text and HTML)
- **Logging System**: Centralized logging for debugging and monitoring
- **REST API**: Flask-based API for real-time detection
- **Cross-Validation**: Built-in cross-validation support
- **Model Versioning**: Track and manage model versions

## Technologies

- Python
- Scikit-learn
- Pandas
- NumPy
- Jupyter Notebook
- Flask (API)
- Matplotlib & Seaborn (Visualization)

## Difficulty Level

Intermediate

## Installation

1. Clone or download this repository
2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Data Preparation
```bash
python scripts/generate_sample_data.py
```

### Model Training
```bash
python scripts/train_model.py
```

### Detection
```bash
python scripts/detect_botnet.py --input data/test_traffic.csv
```

### Hyperparameter Tuning
```bash
python scripts/hyperparameter_tuning.py --data data/processed/training_data.csv
```

### Model Evaluation
```bash
python scripts/model_evaluator.py --data data/processed/training_data.csv --plot --report
```

### Visualization Dashboard
```bash
python scripts/visualization_dashboard.py --data data/processed/training_data.csv
```

### Feature Selection
```bash
python scripts/feature_selector.py
```

### Jupyter Notebook Analysis
```bash
jupyter notebook notebooks/botnet_detection_analysis.ipynb
```

### REST API
```bash
python api/app.py
```
Then access the API at `http://localhost:5000`

## Project Structure

```
botnet-detection/
├── README.md
├── requirements.txt
├── setup.py
├── LICENSE
├── index.html
├── quick_start.py
├── config/
│   └── config.json
├── data/
│   ├── raw/
│   ├── processed/
│   └── models/
├── notebooks/
│   └── botnet_detection_analysis.ipynb
├── scripts/
│   ├── data_processor.py
│   ├── feature_extractor.py
│   ├── feature_selector.py
│   ├── train_model.py
│   ├── hyperparameter_tuning.py
│   ├── detect_botnet.py
│   ├── model_evaluator.py
│   ├── visualization_dashboard.py
│   ├── report_generator.py
│   ├── config_manager.py
│   └── generate_sample_data.py
├── utils/
│   ├── helpers.py
│   └── logger.py
├── api/
│   └── app.py
└── results/
    ├── visualizations/
    └── reports/
```

## Model Performance

The trained model provides detection accuracy metrics including:
- Accuracy
- Precision
- Recall
- F1-Score
- ROC AUC
- Confusion Matrix
- Cross-Validation Scores

## Advanced Usage

### Hyperparameter Tuning
Optimize model hyperparameters for better performance:
```bash
python scripts/hyperparameter_tuning.py --method randomized --n-iter 100
```

### Model Comparison
Compare multiple models side-by-side:
```bash
python scripts/model_evaluator.py --models random_forest gradient_boosting svm --plot --report
```

### Feature Selection
Select the most important features:
```python
from scripts.feature_selector import FeatureSelector
selector = FeatureSelector()
selected_features = selector.select_mutual_info(X, y, k=10)
```

### API Usage
Start the REST API server:
```bash
python api/app.py
```

Example API request:
```bash
curl -X POST http://localhost:5000/detect \
  -H "Content-Type: application/json" \
  -d '{"packet_count": 100, "duration": 10, "total_bytes": 5000}'
```

### Configuration
Modify `config/config.json` to customize:
- Data paths
- Model settings
- Training parameters
- Evaluation metrics
- Logging configuration

## License

This project is for educational purposes only.

## Contact

For questions or support:
- Email: help@rskworld.in, support@rskworld.in
- Phone: +91 93305 39277
- Website: https://rskworld.in

---

© 2025 RSK World. All rights reserved.

