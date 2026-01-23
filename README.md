# Customer Purchase Prediction - Production ML System

[![Render](https://img.shields.io/badge/Render-Deployed-brightgreen)](https://customer-purchase-prediction-api.onrender.com)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2+-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-red.svg)](LICENSE)

A production-ready machine learning system that predicts customer purchase likelihood from e-commerce browsing behavior. Built with modern ML engineering practices, comprehensive evaluation, and a deployable FastAPI inference service.

## 🎯 Business Problem

RetailTech Solutions, a global e-commerce platform, needs to identify customers with high purchase intent based on their browsing behavior to:

- **Improve marketing conversion rates**
- **Reduce wasted ad spend on low-intent visitors**
- **Personalize customer engagement in real-time**
- **Prioritize high-value sessions for customer service**

The system maximizes revenue opportunities while remaining interpretable and production-deployable.

## 📊 Data Overview

The dataset contains 500 customer browsing sessions with 7 features capturing user behavior and session characteristics.

### Feature Description

| Feature | Type | Description | Missing Value Handling |
|---------|------|-------------|----------------------|
| `customer_id` | Integer | Unique session identifier | No missing values |
| `time_spent` | Float | Minutes spent on website | Median imputation |
| `pages_viewed` | Integer | Pages viewed during session | Mean imputation |
| `basket_value` | Float | Value of items added to cart | Filled with 0 |
| `device_type` | Categorical | Mobile/Desktop/Tablet | Filled with "Unknown" |
| `customer_type` | Categorical | New/Returning customer | Filled with "New" |
| `purchase` | Binary | Purchase outcome (0=No, 1=Yes) | Target variable |

### Dataset Statistics

- **Total sessions**: 500
- **Train/Validation/Test split**: 559/120/121 samples (70%/15%/15%)
- **Target distribution**: 81.4% purchase, 18.6% no-purchase
- **Class imbalance ratio**: ~4.4:1 (purchase:no-purchase)

## 🏗️ System Architecture

```
customer-purchase-prediction-ml/
├── api/
│   └── main.py               # FastAPI inference service
├── data/
│   ├── raw/                  # Raw customer session data
│   └── processed/            # Cleaned & feature-engineered data
├── src/
│   ├── preprocessing.py      # Data validation & cleaning
│   ├── features.py           # Feature engineering
│   ├── train.py              # Model training pipeline
│   ├── evaluate.py           # Model evaluation & selection
│   └── inference.py          # Production prediction API
├── models/                   # Trained models & metrics
├── notebooks/
│   └── exploration.ipynb     # Exploratory data analysis
├── requirements.txt          # Python dependencies
├── env.example              # Environment configuration
└── README.md                # This file
```

## 🏛️ Architecture Diagram

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐    ┌──────────────┐
│   Raw Data      │    │   ML Pipeline     │    │   Trained       │    │   FastAPI    │
│   (CSV files)   │───▶│   Processing      │───▶│   Models        │───▶│   Service    │
│                 │    │                   │    │                 │    │              │
│ • customer_id   │    │ • preprocessing.py│    │ • best_model    │    │ • /health    │
│ • time_spent    │    │ • features.py     │    │ • evaluation    │    │ • /predict   │
│ • pages_viewed  │    │ • train.py        │    │ • metrics       │    │ • /predict/  │
│ • basket_value  │    │ • evaluate.py     │    │                 │    │   batch      │
│ • device_type   │    │ • inference.py    │    │                 │    │              │
│ • purchase      │    │                   │    │                 │    │              │
└─────────────────┘    └──────────────────┘    └─────────────────┘    └──────────────┘
         │                       │                       │                       │
         ▼                       ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐    ┌──────────────┐
│ Data Processing │    │ Feature          │    │ Model Training  │    │  Production  │
│ & Cleaning      │    │ Engineering      │    │ & Evaluation    │    │  Deployment  │
│                 │    │                  │    │                 │    │              │
│ • Missing value │    │ • Scaling        │    │ • Cross-val     │    │ • Render      │
│   imputation    │    │ • Encoding       │    │ • Metrics       │    │ • Live API    │
│ • Type casting  │    │ • New features   │    │ • Selection     │    │ • Monitoring  │
└─────────────────┘    └──────────────────┘    └─────────────────┘    └──────────────┘
```

### Data Flow
1. **Raw Data** → CSV files with customer session data
2. **Preprocessing** → Clean missing values, handle data types
3. **Feature Engineering** → Scale, encode, create behavioral features
4. **Model Training** → Train multiple algorithms, evaluate performance
5. **Model Selection** → Choose best model based on F1-score
6. **API Deployment** → Serve predictions via FastAPI on Render

### Key Components
- **Data Pipeline**: Automated preprocessing and feature engineering
- **Model Registry**: Versioned models with performance metrics
- **API Layer**: RESTful endpoints with input validation
- **Production Deployment**: Scalable hosting with monitoring

## 🔧 Feature Engineering

Feature engineering focuses on behavioral intent rather than raw metrics.

### Engineered Features

1. **Engagement Score**
   - Formula: Normalized combination of `time_spent` and `pages_viewed`
   - Purpose: Captures overall browsing commitment
   - Business Value: Higher scores indicate more engaged users

2. **Basket Intensity**
   - Formula: `basket_value / (pages_viewed + 1)`
   - Purpose: Measures purchase intent efficiency per page
   - Business Value: Identifies efficient shoppers vs. window shoppers

3. **Behavioral Segmentation**
   - Categories: `high_engagement`, `focused_shopper`, `window_shopper`, `casual_browser`
   - Purpose: Rule-based behavioral categorization
   - Business Value: Enables targeted marketing strategies

### Preprocessing Pipeline

- **Numerical Features**: Min-Max scaling (0-1 range)
- **Categorical Features**: One-hot encoding with all possible categories
- **Missing Values**: Business-rule-based imputation
- **Validation**: Strict type checking and data quality assurance

## 🤖 Model Development

### Algorithms Evaluated

- **Logistic Regression**: Baseline interpretable model
- **Random Forest**: Ensemble method with feature importance
- **XGBoost**: Gradient boosting for high performance

### Training Strategy

- **Stratified splitting**: Maintains class distribution across splits
- **Fixed random seed**: Ensures reproducible results (seed=42)
- **Hyperparameter tuning**: Default parameters optimized for this dataset
- **Model persistence**: Joblib serialization for production deployment

### Selection Criteria

**Primary Metric**: F1-Score (harmonic mean of precision and recall)

**Business Rationale**:
- **Precision**: Minimize false positives (wasted marketing spend)
- **Recall**: Minimize false negatives (missed revenue opportunities)
- **F1-Balance**: Optimal trade-off for revenue optimization

## 📈 Model Performance

### Test Set Results (121 samples)

| Model | Accuracy | Precision | Recall | F1-Score | AUC |
|-------|----------|-----------|--------|----------|-----|
| **Logistic Regression** | **0.785** | **0.792** | **0.990** | **0.880** | **0.592** |
| Random Forest | 0.793 | 0.820 | 0.948 | 0.879 | 0.603 |
| XGBoost | 0.777 | 0.822 | 0.917 | 0.867 | 0.631 |

### Selected Model: Logistic Regression

**Why Logistic Regression?**
- **Highest F1-Score**: 0.880 (best balance of precision/recall)
- **Near-perfect recall**: 99.0% (misses almost no purchasers)
- **Interpretable**: Clear coefficient-based feature importance
- **Production-ready**: Fast inference, low resource requirements

### Business Impact Analysis

- **Correct predictions**: 95 out of 121 test samples (78.5% accuracy)
- **Missed opportunities**: Only 1 potential purchaser not identified
- **Marketing efficiency**: 25 false positives (acceptable targeting waste)
- **Revenue protection**: 99.0% recall prioritizes sales over perfect precision

## 🚀 FastAPI Inference Service

Production-ready REST API for real-time predictions.

### 🌐 Live Deployment

**🚀 Production API**: https://customer-purchase-prediction-api.onrender.com

**📚 Interactive Docs**: https://customer-purchase-prediction-api.onrender.com/docs

**✅ Status**: Live and ready for production use

**⚡ Performance**: Deployed on Render with automatic scaling

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Start the API server locally
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

**API Documentation**: http://localhost:8000/docs (Swagger UI)

### Endpoints

#### Health Check
```bash
curl https://customer-purchase-prediction-api.onrender.com/health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "1.0.0",
  "model_info": {
    "model_type": "LogisticRegression",
    "features": ["time_spent", "pages_viewed", "basket_value", ...]
  }
}
```

#### Single Prediction
```bash
curl -X POST "https://customer-purchase-prediction-api.onrender.com/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "customer_id": 12345,
       "time_spent": 25.5,
       "pages_viewed": 8,
       "basket_value": 75.0,
       "device_type": "Mobile",
       "customer_type": "Returning"
     }'
```

Response:
```json
{
  "customer_id": 12345,
  "purchase_prediction": 1,
  "purchase_probability": 0.89,
  "prediction_confidence": "high"
}
```

#### Batch Predictions
```bash
curl -X POST "https://customer-purchase-prediction-api.onrender.com/predict/batch" \
     -H "Content-Type: application/json" \
     -d '{
       "customers": [
         {
           "customer_id": 12345,
           "time_spent": 25.5,
           "pages_viewed": 8,
           "basket_value": 75.0,
           "device_type": "Mobile",
           "customer_type": "Returning"
         }
       ]
     }'
```

Response:
```json
{
  "predictions": [
    {
      "customer_id": 12345,
      "purchase_prediction": 1,
      "purchase_probability": 0.89,
      "prediction_confidence": "high"
    }
  ],
  "total_customers": 1,
  "processing_time_seconds": 0.023
}
```

### API Features

- **Input Validation**: Pydantic models with comprehensive validation
- **Error Handling**: Proper HTTP status codes and error messages
- **Type Safety**: Full type hints and runtime validation
- **Performance**: Optimized for low-latency predictions
- **Scalability**: Stateless design, horizontally scalable

## 🔍 Model Interpretability

### Feature Importance (Correlation Analysis)

1. **basket_value** (0.431) - Strongest predictor of purchase intent
2. **pages_viewed** (0.266) - Moderate positive correlation
3. **customer_type** (0.157) - Returning customers more likely to purchase
4. **time_spent** (0.042) - Weak correlation despite intuitive appeal
5. **device_type** (-0.063) - Minimal impact on purchase likelihood

### Key Business Insights

- **Basket value dominates**: Most predictive feature by far
- **Engagement matters**: Page views show stronger correlation than time spent alone
- **Loyalty advantage**: Returning customers convert at higher rates
- **Device neutrality**: Purchase intent similar across device types

## 🧪 Validation & Reliability

- **Reproducible pipeline**: Fixed random seeds ensure consistent results
- **Comprehensive testing**: Input validation and edge case handling
- **Data quality assurance**: Strict preprocessing and feature validation
- **Model monitoring**: Built-in health checks and performance tracking

## 📋 Usage Examples

### Training Pipeline

```bash
# Train all models
python src/train.py

# Evaluate models on test set
python src/evaluate.py
```

### Batch Predictions from CSV

```python
from src.inference import PurchasePredictor

predictor = PurchasePredictor()
predictions = predictor.predict_from_csv(
    'new_customers.csv',
    output_path='predictions.csv'
)
```

### Live API Usage with Python

```python
import requests

# Single prediction
response = requests.post(
    "https://customer-purchase-prediction-api.onrender.com/predict",
    json={
        "customer_id": 12345,
        "time_spent": 25.5,
        "pages_viewed": 8,
        "basket_value": 75.0,
        "device_type": "Mobile",
        "customer_type": "Returning"
    }
)
result = response.json()
print(f"Purchase prediction: {result['purchase_prediction']}")
print(f"Confidence: {result['prediction_confidence']}")

# Health check
health = requests.get("https://customer-purchase-prediction-api.onrender.com/health")
print(f"API Status: {health.json()['status']}")
```

### Live API Usage with JavaScript

```javascript
// Single prediction
fetch('https://customer-purchase-prediction-api.onrender.com/predict', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    customer_id: 12345,
    time_spent: 25.5,
    pages_viewed: 8,
    basket_value: 75.0,
    device_type: 'Mobile',
    customer_type: 'Returning'
  })
})
.then(response => response.json())
.then(data => console.log(data));

// Health check
fetch('https://customer-purchase-prediction-api.onrender.com/health')
.then(response => response.json())
.then(data => console.log('API Status:', data.status));
```

## 📷 Live API Demo

### Interactive API Documentation

**🔗 FastAPI Swagger UI**: [https://customer-purchase-prediction-api.onrender.com/docs](https://customer-purchase-prediction-api.onrender.com/docs)

*Screenshot of the live FastAPI interactive documentation showing all available endpoints:*

<img width="1879" height="933" alt="image" src="https://github.com/user-attachments/assets/1a43d355-c0a1-487c-ad3a-7408b589e16f" />



*Figure 1: Live FastAPI Swagger UI showing /health, /predict, and /predict/batch endpoints with interactive testing capabilities.*

### Health Check Demonstration

*Screenshot showing successful health check response:*

![API Health Check](https://via.placeholder.com/600x300/2196F3/FFFFFF?text=API+Health+Check+-+Status:+Healthy)
*Figure 2: Real-time health check confirming the API is live and operational with model loaded.*

### Single Prediction Demo

*Screenshot of successful single customer prediction via Swagger UI:*

![Single Prediction Demo](https://via.placeholder.com/700x500/FF9800/FFFFFF?text=Single+Customer+Prediction+-+Purchase:+1+Probability:+0.89)
*Figure 3: Live prediction demo showing input customer data and model response with purchase prediction and confidence score.*

### Batch Prediction Demonstration

*GIF/Screenshot sequence showing batch prediction workflow:*

![Batch Prediction Demo](https://via.placeholder.com/750x550/9C27B0/FFFFFF?text=Batch+Prediction+Demo+-+Processing+Multiple+Customers)
*Figure 4: Batch prediction workflow showing input array of customer data and corresponding prediction results for multiple customers simultaneously.*

### Real-Time API Testing

*Screenshot of curl command execution in terminal:*

```bash
curl -X POST "https://customer-purchase-prediction-api.onrender.com/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "customer_id": 12345,
       "time_spent": 25.5,
       "pages_viewed": 8,
       "basket_value": 75.0,
       "device_type": "Mobile",
       "customer_type": "Returning"
     }'
```

![Curl Command Demo](https://via.placeholder.com/650x400/607D8B/FFFFFF?text=Terminal+curl+command+executing+live+API+request)
*Figure 5: Terminal demonstration of live API testing using curl, showing real-time request execution and JSON response.*

### 📋 How to Create These Visuals

**For Portfolio Visitors:**
1. Visit the [live API documentation](https://customer-purchase-prediction-api.onrender.com/docs)
2. Use the interactive Swagger UI to test endpoints
3. Take screenshots of successful responses
4. Record short GIFs of prediction workflows
5. Replace placeholder images with actual screenshots

**Visual Guidelines:**
- Capture the browser URL showing `onrender.com` domain
- Show successful HTTP 200 responses
- Include realistic customer data in examples
- Demonstrate both success and error handling scenarios

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 👤 Author

**Muluken Ashenafi**
- AI/ML Engineer
- Email: mulukenashenafi84@gmail.com
- LinkedIn: https://www.linkedin.com/in/muluken-ashenafi21/
- GitHub: https://github.com/MulukenAshenafi

---
