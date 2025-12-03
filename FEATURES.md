# Botnet Detection Project - Features List

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

## Core Features

1. **Network Traffic Analysis**
   - Data loading and preprocessing
   - Statistical analysis
   - Data cleaning utilities

2. **Botnet Pattern Recognition**
   - Multiple ML algorithms (Random Forest, Gradient Boosting, SVM, Logistic Regression)
   - Pattern detection in network traffic
   - Real-time detection capabilities

3. **DNS Query Analysis**
   - DNS query rate calculation
   - Suspicious DNS pattern detection
   - Query frequency analysis

4. **Classification Model Training**
   - Automated model training pipeline
   - Multiple algorithm support
   - Model persistence and loading

5. **Detection Accuracy Metrics**
   - Accuracy, Precision, Recall, F1-Score
   - ROC AUC calculation
   - Confusion matrix generation

## Advanced Features

### 1. Hyperparameter Tuning (`scripts/hyperparameter_tuning.py`)
- **GridSearchCV**: Exhaustive search over parameter grid
- **RandomizedSearchCV**: Randomized search for faster optimization
- Supports Random Forest and Gradient Boosting
- Saves tuned models automatically
- Configurable number of iterations

**Usage:**
```bash
python scripts/hyperparameter_tuning.py --method randomized --n-iter 100
```

### 2. Model Evaluation & Comparison (`scripts/model_evaluator.py`)
- **Comprehensive Metrics**: Accuracy, Precision, Recall, F1-Score, ROC AUC
- **Cross-Validation**: Built-in 5-fold cross-validation
- **Model Comparison**: Side-by-side comparison of multiple models
- **Visualization**: Automatic generation of comparison plots
- **Report Generation**: Detailed text reports

**Usage:**
```bash
python scripts/model_evaluator.py --models random_forest gradient_boosting --plot --report
```

### 3. Visualization Dashboard (`scripts/visualization_dashboard.py`)
- **Data Overview**: Class distribution, feature distributions, correlation matrix
- **Feature Analysis**: Detailed feature distribution plots
- **Traffic Patterns**: Network traffic pattern visualization
- **Automatic Export**: Saves all visualizations as high-resolution PNG files

**Usage:**
```bash
python scripts/visualization_dashboard.py --data data/processed/training_data.csv
```

### 4. Feature Selection (`scripts/feature_selector.py`)
- **K-Best Selection**: Statistical test-based feature selection
- **RFE (Recursive Feature Elimination)**: Model-based feature elimination
- **Mutual Information**: Information-theoretic feature selection
- **Model-based Selection**: Feature importance from trained models
- **Multiple Scoring Functions**: f_classif, chi2, mutual_info_classif

**Usage:**
```python
from scripts.feature_selector import FeatureSelector
selector = FeatureSelector()
selected = selector.select_mutual_info(X, y, k=10)
```

### 5. Configuration Management (`scripts/config_manager.py`)
- **Centralized Configuration**: JSON-based configuration system
- **Easy Access**: Dot-notation for nested configuration
- **Default Values**: Automatic fallback to defaults
- **Project Info**: Built-in project metadata

**Configuration File:** `config/config.json`

### 6. Report Generation (`scripts/report_generator.py`)
- **Training Reports**: Detailed training results
- **Detection Reports**: Detection summary and statistics
- **HTML Reports**: Beautiful HTML summary reports
- **Automatic Timestamping**: Unique filenames with timestamps

**Usage:**
```python
from scripts.report_generator import ReportGenerator
generator = ReportGenerator()
generator.generate_training_report(training_results)
```

### 7. Logging System (`utils/logger.py`)
- **Centralized Logging**: Single logger instance
- **File & Console**: Dual output (file and console)
- **Configurable Levels**: DEBUG, INFO, WARNING, ERROR
- **Automatic Directory Creation**: Creates log directories automatically

**Usage:**
```python
from utils.logger import get_logger
logger = get_logger()
logger.info("Training started")
```

### 8. REST API (`api/app.py`)
- **Flask-based API**: RESTful endpoints
- **Single Detection**: `/detect` endpoint for single records
- **Batch Detection**: `/detect/batch` endpoint for multiple records
- **Health Check**: `/health` endpoint for monitoring
- **CORS Support**: Cross-origin resource sharing enabled

**Endpoints:**
- `GET /` - API information
- `GET /health` - Health check
- `POST /detect` - Single detection
- `POST /detect/batch` - Batch detection

**Usage:**
```bash
python api/app.py
# API runs on http://localhost:5000
```

### 9. Cross-Validation Support
- **Built-in CV**: Integrated in model evaluator
- **Stratified K-Fold**: Maintains class distribution
- **Configurable Folds**: Adjustable number of CV folds
- **Score Aggregation**: Mean and standard deviation reporting

### 10. Enhanced Feature Extraction
- **Automatic Feature Engineering**: Rate calculations, normalized features
- **Statistical Features**: Mean, std-based normalization
- **Traffic Metrics**: Packets/sec, bytes/sec, connections/sec
- **DNS Metrics**: DNS query rates

### 11. Data Processing Pipeline
- **Data Cleaning**: Missing value handling, duplicate removal
- **Feature Extraction**: Automatic feature engineering
- **Data Validation**: Type checking and validation

### 12. Model Persistence
- **Pickle Serialization**: Model saving and loading
- **Feature Name Storage**: JSON-based feature name persistence
- **Model Metadata**: Training results and parameters saved

## Project Structure Enhancements

- **Configuration Directory**: `config/` for centralized settings
- **API Directory**: `api/` for REST API endpoints
- **Results Directory**: `results/` for outputs (visualizations, reports)
- **Logs Directory**: `logs/` for log files

## Dependencies Added

- **Flask**: Web framework for API
- **Flask-CORS**: Cross-origin resource sharing support

## Usage Examples

### Complete Workflow
```bash
# 1. Generate sample data
python scripts/generate_sample_data.py

# 2. Train models
python scripts/train_model.py

# 3. Tune hyperparameters (optional)
python scripts/hyperparameter_tuning.py

# 4. Evaluate models
python scripts/model_evaluator.py --plot --report

# 5. Generate visualizations
python scripts/visualization_dashboard.py

# 6. Run detection
python scripts/detect_botnet.py --input data/processed/training_data.csv

# 7. Start API (optional)
python api/app.py
```

## Contact

**RSK World**
- Founder: Molla Samser
- Designer & Tester: Rima Khatun
- Email: help@rskworld.in, support@rskworld.in
- Phone: +91 93305 39277
- Website: https://rskworld.in

---

© 2025 RSK World. All rights reserved.

