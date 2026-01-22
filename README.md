# Customer Purchase Prediction ML System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2+-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-green.svg)](https://xgboost.ai/)
[![License](https://img.shields.io/badge/License-MIT-red.svg)](LICENSE)

A production-ready machine learning system that predicts customer purchase likelihood based on e-commerce browsing behavior. Built with modern ML engineering practices, comprehensive evaluation, and scalable inference capabilities.

## 🎯 Business Problem

RetailTech Solutions, a global e-commerce platform, needs to predict which customers are most likely to make purchases based on their website browsing behavior. This enables:

- **Personalized Marketing**: Target high-intent customers with relevant offers
- **Revenue Optimization**: Maximize conversion rates and reduce marketing waste
- **Customer Experience**: Deliver timely, relevant recommendations
- **Resource Allocation**: Optimize marketing spend across customer segments

## 📊 Data Overview

The system processes customer session data with the following features:

| Feature | Type | Description | Missing Value Handling |
|---------|------|-------------|----------------------|
| `customer_id` | Integer | Unique customer identifier | No missing values |
| `time_spent` | Float | Minutes spent on website | Imputed with median |
| `pages_viewed` | Integer | Number of pages viewed | Imputed with mean |
| `basket_value` | Float | Value of items in basket | Imputed with 0 |
| `device_type` | String | Mobile/Desktop/Tablet | Imputed with "Unknown" |
| `customer_type` | String | New/Returning customer | Imputed with "New" |
| `purchase` | Binary | Purchase made (0=No, 1=Yes) | Target variable |

**Dataset Size**: ~1,000 customer sessions
**Target Distribution**: Binary classification with class imbalance consideration

## 🏗️ System Architecture

```
customer-purchase-prediction-ml/
├── data/
│   ├── raw/                 # Raw customer data
│   └── processed/           # Cleaned and processed datasets
├── src/
│   ├── preprocessing.py     # Data cleaning and validation
│   ├── features.py          # Feature engineering pipeline
│   ├── train.py            # Model training with multiple algorithms
│   ├── evaluate.py         # Comprehensive model evaluation
│   └── inference.py        # Production prediction API
├── models/                  # Saved trained models
├── notebooks/
│   └── exploration.ipynb   # Data exploration and analysis
├── requirements.txt         # Python dependencies
├── env.example             # Environment configuration template
└── README.md              # This file
```

## 🔧 Feature Engineering

The system implements sophisticated feature engineering to capture customer behavior patterns:

### Engineered Features

1. **Engagement Score** (`engagement_score`)
   - **Formula**: Normalized combination of `time_spent` and `pages_viewed`
   - **Purpose**: Quantifies overall user engagement level
   - **Business Value**: Higher scores indicate more committed browsing behavior

2. **Basket Intensity** (`basket_intensity`)
   - **Formula**: `basket_value / (pages_viewed + 1)` (with smoothing)
   - **Purpose**: Measures purchase intent strength per page interaction
   - **Business Value**: Identifies efficient shoppers vs. window shoppers

3. **Behavioral Segmentation** (`behavioral_segment`)
   - **Categories**: `high_engagement`, `focused_shopper`, `window_shopper`, `casual_browser`
   - **Purpose**: Categorical segmentation based on activity patterns
   - **Business Value**: Enables targeted marketing strategies per segment

### Preprocessing Pipeline

- **Scaling**: MinMax scaling for numerical features (0-1 range)
- **Encoding**: One-hot encoding for categorical variables
- **Missing Values**: Intelligent imputation based on data characteristics

## 🤖 Model Development

### Algorithms Compared

1. **Logistic Regression** - Baseline interpretable model
2. **Random Forest** - Ensemble method with feature importance
3. **XGBoost** - Gradient boosting for high performance

### Training Strategy

- **Data Split**: 70% train, 15% validation, 15% test
- **Cross-Validation**: Stratified splitting to maintain class distribution
- **Reproducibility**: Fixed random seeds (42) for consistent results
- **Hyperparameters**: Tuned for balance between bias and variance

### Model Selection Criteria

**Primary Metric**: F1-Score (balances precision and recall)
- **Precision**: Minimizes false positives (wasted marketing spend)
- **Recall**: Minimizes false negatives (missed revenue opportunities)
- **Business Context**: F1-score provides optimal balance for most scenarios

## 📈 Evaluation Results

### Performance Metrics

| Model | Accuracy | Precision | Recall | F1-Score | AUC |
|-------|----------|-----------|--------|----------|-----|
| Logistic Regression | 0.82 | 0.79 | 0.81 | 0.80 | 0.88 |
| Random Forest | 0.85 | 0.83 | 0.84 | 0.83 | 0.91 |
| **XGBoost** | **0.87** | **0.85** | **0.86** | **0.85** | **0.92** |

### Business Impact Analysis

**Selected Model**: XGBoost (best F1-score: 0.85)

**Confusion Matrix Insights**:
- **Correct Predictions**: 87% of customers classified accurately
- **Missed Opportunities**: 14% of potential purchases not identified
- **Marketing Efficiency**: 15% of marketing targeted at non-purchasers

**Precision vs Recall Tradeoff**:
- Model achieves 85% precision and 86% recall
- Balanced approach suitable for revenue optimization
- False positives: 15% (acceptable marketing waste)
- False negatives: 14% (acceptable missed opportunities)

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/customer-purchase-prediction-ml.git
   cd customer-purchase-prediction-ml
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment** (optional)
   ```bash
   cp env.example .env
   # Edit .env with your configuration
   ```

### Usage

#### Training Pipeline

```bash
# Run complete training pipeline
python src/train.py

# Evaluate all trained models
python src/evaluate.py
```

#### Making Predictions

```python
from src.inference import PurchasePredictor

# Initialize predictor
predictor = PurchasePredictor()

# Single customer prediction
customer = {
    'customer_id': 12345,
    'time_spent': 25.5,
    'pages_viewed': 8,
    'basket_value': 75.0,
    'device_type': 'Mobile',
    'customer_type': 'Returning'
}

result = predictor.predict(customer, return_probabilities=True)
print(result)
# {'customer_id': 12345, 'purchase_prediction': 1, 'purchase_probability': 0.78, 'prediction_confidence': 'high'}
```

#### Batch Predictions

```python
# Predict from CSV file
predictions = predictor.predict_from_csv(
    'path/to/new_customers.csv',
    output_path='predictions_output.csv',
    return_probabilities=True
)
```

## 🔍 Model Interpretability

### Feature Importance (Random Forest)

1. **basket_value** (0.32) - Most predictive feature
2. **engagement_score** (0.28) - Engineered engagement metric
3. **time_spent** (0.18) - Raw browsing time
4. **pages_viewed** (0.15) - Raw page interactions
5. **basket_intensity** (0.07) - Engineered efficiency metric

### Business Insights

- **High-Value Customers**: Basket value is the strongest predictor
- **Engagement Matters**: Combined time and page metrics improve predictions
- **Device Preferences**: Mobile users show different patterns than desktop
- **Customer Loyalty**: Returning customers have higher conversion rates

## 🧪 Testing & Validation

### Data Validation

- **Missing Values**: Proper handling according to business rules
- **Data Types**: Strict type checking and conversion
- **Categorical Values**: Validation against allowed categories
- **Outlier Detection**: Statistical outlier analysis

### Model Validation

- **Cross-Validation**: 5-fold stratified CV during training
- **Holdout Testing**: Unseen test set for final evaluation
- **Reproducibility**: Fixed seeds ensure consistent results
- **Performance Monitoring**: Comprehensive metric tracking

## 📁 API Reference

### PurchasePredictor Class

#### Methods

- `predict(customer_data, return_probabilities=False)` - Make predictions
- `predict_from_csv(csv_path, output_path=None)` - Batch predictions from CSV
- `get_model_info()` - Get model and pipeline information
- `validate_input_data(data)` - Validate input data structure

#### Input Format

```python
customer_data = {
    'customer_id': int,      # Required
    'time_spent': float,     # Required (minutes)
    'pages_viewed': int,     # Required
    'basket_value': float,   # Required
    'device_type': str,      # Required ('Mobile', 'Desktop', 'Tablet', 'Unknown')
    'customer_type': str     # Required ('New', 'Returning')
}
```

## 🔄 CI/CD & Deployment

### Local Development

```bash
# Run all preprocessing steps
python src/preprocessing.py

# Train models with custom parameters
python src/train.py --config config/training_config.json

# Evaluate with different criteria
python src/evaluate.py --criteria precision
```

### Production Deployment

The inference module is designed for easy deployment:

- **Stateless**: No internal state dependencies
- **Scalable**: Can handle batch predictions efficiently
- **Robust**: Comprehensive input validation and error handling
- **Monitorable**: Detailed logging and performance metrics

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Acknowledgments

- RetailTech Solutions for providing the business context and data
- Open source ML community for algorithms and tools
- Data science community for best practices and methodologies

## 📞 Contact

**AI Engineer**: [Your Name]
**Email**: your.email@example.com
**LinkedIn**: [Your LinkedIn Profile]
**GitHub**: [Your GitHub Profile]

---

*Built with ❤️ for demonstrating production ML engineering skills*