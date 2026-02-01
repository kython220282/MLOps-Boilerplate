# ML Service Framework - Complete Documentation

## Table of Contents

1. [Introduction](#introduction)
2. [Installation Guide](#installation-guide)
3. [Quick Start Tutorial](#quick-start-tutorial)
4. [Supported Machine Learning Models](#supported-machine-learning-models)
5. [Step-by-Step Usage Guide](#step-by-step-usage-guide)
6. [Advanced Features](#advanced-features)
7. [Advantages](#advantages)
8. [Current Limitations](#current-limitations)
9. [Roadmap](#roadmap)
10. [Troubleshooting](#troubleshooting)

---

## Introduction

ML Service Framework is a production-ready, end-to-end machine learning framework designed to accelerate ML project development from experimentation to deployment. It provides a complete, opinionated structure for building, training, deploying, and monitoring machine learning models at scale.

### Who Should Use This Framework?

- **Data Scientists** who want to focus on modeling rather than infrastructure
- **ML Engineers** building production ML systems
- **Teams** looking for standardized ML project structure
- **Startups** needing rapid ML prototyping and deployment
- **Organizations** requiring MLOps best practices out-of-the-box

---

## Installation Guide

### System Requirements

- **Python:** 3.9, 3.10, or 3.11
- **Operating System:** Windows, Linux, or macOS
- **RAM:** Minimum 4GB (8GB+ recommended for larger datasets)
- **Disk Space:** 2GB for framework + space for your data and models

### Method 1: Install from PyPI (Recommended)

```bash
# Install the framework
pip install ml-service-framework

# Verify installation
ml-create-project --help
```

### Method 2: Install from Source

```bash
# Clone the repository
git clone https://github.com/yourusername/ml-service-framework.git
cd ml-service-framework

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install in development mode
pip install -e .
```

### Method 3: Using Docker

```bash
# Pull the Docker image
docker pull yourusername/ml-service-framework:latest

# Run container
docker run -it -p 8000:8000 ml-service-framework
```

---

## Quick Start Tutorial

### Step 1: Create Your First Project

```bash
# Create a new ML project
ml-create-project my-first-ml-project

# Navigate to the project directory
cd my-first-ml-project
```

### Step 2: Set Up Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 3: Configure Your Project

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your settings
notepad .env  # Windows
nano .env     # Linux/Mac
```

### Step 4: Prepare Your Data

```bash
# Create a data directory and add your CSV file
mkdir -p data/raw
# Copy your training data to data/raw/train.csv
```

**Example data format (train.csv):**
```csv
feature1,feature2,feature3,target
1.2,3.4,5.6,0
2.3,4.5,6.7,1
3.4,5.6,7.8,0
...
```

### Step 5: Configure Training

Edit `config/training_config.json`:

```json
{
  "data_source": {
    "type": "file",
    "path": "data/raw/train.csv"
  },
  "target_column": "target",
  "model": {
    "type": "random_forest",
    "model_type": "classifier",
    "n_estimators": 100,
    "max_depth": 10,
    "random_state": 42
  },
  "data_processing": {
    "missing_value_strategy": "mean",
    "scaling_method": "standard",
    "test_size": 0.2,
    "random_state": 42
  },
  "cross_validation": {
    "type": "kfold",
    "n_splits": 5,
    "scoring": "accuracy"
  },
  "task_type": "classification",
  "output_path": "models/model.joblib",
  "run_cross_validation": true,
  "random_state": 42
}
```

### Step 6: Train Your Model

```bash
# Train using the configuration file
ml-train --config config/training_config.json
```

**Expected Output:**
```
2026-02-01 10:00:00 - INFO - Starting training pipeline...
2026-02-01 10:00:01 - INFO - Loaded data with shape: (1000, 10)
2026-02-01 10:00:02 - INFO - Data preprocessing completed
2026-02-01 10:00:05 - INFO - Model training completed. Metrics: {'train_score': 0.95, 'val_score': 0.92}
2026-02-01 10:00:06 - INFO - Model saved to models/model.joblib
2026-02-01 10:00:06 - INFO - Training pipeline completed successfully
```

### Step 7: Run Inference

```bash
# Prepare test data (test.csv)
ml-inference --model-path models/model.joblib \
             --input-path data/raw/test.csv \
             --output-path predictions.csv
```

### Step 8: Serve Model via API

```bash
# Start the API server
ml-serve --model-path models/model.joblib --port 8000
```

**Test the API:**
```bash
# Health check
curl http://localhost:8000/health

# Make a prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": {"feature1": 1.2, "feature2": 3.4, "feature3": 5.6}}'
```

---

## Supported Machine Learning Models

### 1. Random Forest

**Use Cases:**
- Classification tasks (fraud detection, customer churn)
- Regression tasks (price prediction, demand forecasting)
- Feature importance analysis
- Handling mixed data types

**Configuration Example:**
```json
{
  "model": {
    "type": "random_forest",
    "model_type": "classifier",  // or "regressor"
    "n_estimators": 100,
    "max_depth": 10,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "random_state": 42
  }
}
```

**Advantages:**
- Works well out-of-the-box
- Handles non-linear relationships
- Robust to outliers
- Built-in feature importance

**Best For:**
- Small to medium datasets (< 100K rows)
- Interpretability required
- Quick prototyping

---

### 2. XGBoost

**Use Cases:**
- Kaggle competitions
- Imbalanced classification
- High-performance predictions
- Complex pattern recognition

**Configuration Example:**
```json
{
  "model": {
    "type": "xgboost",
    "model_type": "classifier",
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42
  }
}
```

**Advantages:**
- State-of-the-art performance
- Handles missing values natively
- Built-in regularization
- Fast training with GPU support

**Best For:**
- Large datasets (> 100K rows)
- When highest accuracy is needed
- Structured/tabular data

---

### 3. Custom Models (Extensible)

You can add any scikit-learn compatible model or custom models:

**Example: Adding Logistic Regression**

Create `ml_service/machine_learning/custom_models.py`:

```python
from ml_service.machine_learning.model import BaseModel
from sklearn.linear_model import LogisticRegression

class LogisticRegressionModel(BaseModel):
    def build_model(self):
        self.model = LogisticRegression(
            max_iter=self.config.get('max_iter', 1000),
            random_state=self.config.get('random_state', 42)
        )
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        if self.model is None:
            self.build_model()
        
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        metrics = {'train_score': self.model.score(X_train, y_train)}
        if X_val is not None:
            metrics['val_score'] = self.model.score(X_val, y_val)
        
        return metrics
    
    def predict(self, X):
        return self.model.predict(X)
```

**Register in ModelFactory:**

Edit `ml_service/machine_learning/model.py`:

```python
from ml_service.machine_learning.custom_models import LogisticRegressionModel

class ModelFactory:
    @staticmethod
    def create_model(model_type: str, config: Optional[Dict[str, Any]] = None) -> BaseModel:
        models = {
            'random_forest': RandomForestModel,
            'xgboost': XGBoostModel,
            'logistic_regression': LogisticRegressionModel,  # Add this
        }
        # ... rest of the code
```

---

## Step-by-Step Usage Guide

### Scenario 1: Binary Classification with CSV Data

**Goal:** Predict customer churn (Yes/No)

**Step 1:** Prepare data
```csv
customer_id,age,tenure,monthly_charges,total_charges,churn
C001,25,12,50.5,606.0,No
C002,45,24,80.2,1924.8,Yes
...
```

**Step 2:** Configuration
```json
{
  "data_source": {"type": "file", "path": "data/churn.csv"},
  "target_column": "churn",
  "model": {
    "type": "random_forest",
    "model_type": "classifier",
    "n_estimators": 200
  },
  "task_type": "classification"
}
```

**Step 3:** Train
```bash
ml-train --config config/churn_config.json
```

---

### Scenario 2: Regression with Database

**Goal:** Predict house prices from PostgreSQL database

**Step 1:** Setup database connection in `.env`:
```bash
DB_TYPE=postgresql
DB_HOST=localhost
DB_PORT=5432
DB_NAME=real_estate
DB_USER=ml_user
DB_PASSWORD=your_password
```

**Step 2:** Configuration
```json
{
  "data_source": {
    "type": "database",
    "connector_type": "postgresql",
    "connection_config": {
      "host": "localhost",
      "port": 5432,
      "database": "real_estate",
      "user": "ml_user",
      "password": "your_password"
    },
    "query": "SELECT * FROM house_sales WHERE sale_date >= '2023-01-01'"
  },
  "target_column": "price",
  "model": {
    "type": "xgboost",
    "model_type": "regressor",
    "n_estimators": 300,
    "learning_rate": 0.05
  },
  "task_type": "regression"
}
```

**Step 3:** Train
```bash
ml-train --config config/house_price_config.json
```

---

### Scenario 3: Training with Cloud Data (S3)

**Goal:** Train model using data stored in AWS S3

**Step 1:** Configure AWS credentials in `.env`:
```bash
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_REGION=us-east-1
S3_BUCKET_NAME=my-ml-data
```

**Step 2:** Configuration
```json
{
  "data_source": {
    "type": "object_storage",
    "connector_type": "s3",
    "connection_config": {
      "bucket": "my-ml-data",
      "access_key_id": "your_access_key",
      "secret_access_key": "your_secret_key",
      "region": "us-east-1"
    },
    "remote_path": "datasets/training_data.csv",
    "local_path": "data/downloaded_data.csv"
  },
  "target_column": "target"
}
```

---

### Scenario 4: Hyperparameter Tuning with Optuna

**Step 1:** Python script for tuning
```python
from ml_service.machine_learning.hyperparameter_tuning import (
    HyperparameterTuner, 
    create_optuna_param_space_rf
)
from ml_service.machine_learning.data_processor import DataProcessor
from sklearn.ensemble import RandomForestClassifier

# Load and preprocess data
processor = DataProcessor()
df = processor.load_data("data/train.csv")
X_train, X_test, y_train, y_test = processor.preprocess_pipeline(df, "target")

# Hyperparameter tuning
tuner = HyperparameterTuner(method='optuna', n_trials=100)
results = tuner.optuna_search(
    RandomForestClassifier,
    X_train,
    y_train,
    create_optuna_param_space_rf,
    scoring='accuracy'
)

print(f"Best parameters: {results['best_params']}")
print(f"Best score: {results['best_score']}")

# Train final model with best parameters
best_model = results['best_model']
```

---

### Scenario 5: Experiment Tracking with MLflow

**Step 1:** Start MLflow server
```bash
mlflow server --host 0.0.0.0 --port 5000
```

**Step 2:** Python script with tracking
```python
from ml_service.machine_learning.experiment_tracking import MLflowTracker
from ml_service.machine_learning.training_pipeline import TrainingPipeline

# Initialize tracker
tracker = MLflowTracker(
    tracking_uri="http://localhost:5000",
    experiment_name="customer_churn"
)

# Load config and train
config = {...}  # Your training config
pipeline = TrainingPipeline(config)

# Run with tracking
with tracker.start_run(run_name="rf_baseline"):
    metrics = pipeline.run_pipeline()
    tracker.log_params(config['model'])
    tracker.log_metrics(metrics['test'])
    tracker.log_model(pipeline.model, registered_model_name="churn_model")
```

**Step 3:** View experiments
```bash
# Open browser to http://localhost:5000
```

---

### Scenario 6: Model Deployment with Docker

**Step 1:** Build Docker image
```bash
docker build -t my-ml-service .
```

**Step 2:** Run with docker-compose
```bash
docker-compose up -d
```

This starts:
- ML API server (port 8000)
- PostgreSQL database (port 5432)
- MongoDB (port 27017)
- MLflow (port 5000)
- Prometheus (port 9091)
- Grafana (port 3000)

**Step 3:** Deploy to production
```bash
# Push to registry
docker tag my-ml-service:latest registry.example.com/my-ml-service:v1.0
docker push registry.example.com/my-ml-service:v1.0

# Deploy to Kubernetes (example)
kubectl apply -f deployment.yaml
```

---

### Scenario 7: Monitoring and Data Drift Detection

**Step 1:** Set up monitoring
```python
from ml_service.monitoring import ModelMonitor, DataDriftDetector
import pandas as pd

# Initialize monitoring
monitor = ModelMonitor(log_dir="monitoring/logs")

# Log predictions
monitor.log_prediction(
    model_name="churn_model",
    model_version="v1",
    input_data={"age": 35, "tenure": 12},
    prediction="Yes",
    latency=0.05
)
```

**Step 2:** Detect data drift
```python
# Load reference data (training data)
reference_data = pd.read_csv("data/train.csv")

# Load current production data
current_data = pd.read_csv("monitoring/production_data.csv")

# Initialize drift detector
drift_detector = DataDriftDetector(
    reference_data=reference_data,
    drift_threshold=0.1
)

# Detect drift
drift_results = drift_detector.detect_all_features_drift(current_data)

# Generate report
drift_detector.generate_drift_report_evidently(
    current_data,
    output_path="monitoring/drift_report.html"
)
```

---

## Advanced Features

### 1. Custom Data Preprocessing

Create custom preprocessing steps:

```python
from ml_service.machine_learning.data_processor import DataProcessor

class CustomDataProcessor(DataProcessor):
    def create_features(self, df):
        """Custom feature engineering."""
        df = super().create_features(df)
        
        # Add custom features
        df['age_tenure_ratio'] = df['age'] / (df['tenure'] + 1)
        df['avg_monthly_charge'] = df['total_charges'] / (df['tenure'] + 1)
        
        return df
```

### 2. Model Registry Usage

```python
from ml_service.machine_learning.model_registry import ModelRegistry

# Initialize registry
registry = ModelRegistry(registry_path="model_registry")

# Register a model
version = registry.register_model(
    model_name="churn_predictor",
    model_path="models/model.joblib",
    metadata={
        "metrics": {"accuracy": 0.92, "f1_score": 0.89},
        "training_date": "2026-02-01",
        "dataset_size": 10000
    },
    tags={"environment": "production", "version": "1.0"}
)

# Load model from registry
model = registry.load_model("churn_predictor", stage="Production")

# List all versions
versions = registry.list_versions("churn_predictor")
```

### 3. Batch Inference at Scale

```python
from ml_service.applications.inference import InferenceService
import pandas as pd

# Initialize service
service = InferenceService("models/model.joblib")

# Process large file in chunks
chunk_size = 10000
output_file = "predictions_large.csv"

for chunk in pd.read_csv("data/large_dataset.csv", chunksize=chunk_size):
    predictions = service.predict(chunk)
    chunk['predictions'] = predictions
    
    # Append to output
    chunk.to_csv(output_file, mode='a', header=not Path(output_file).exists(), index=False)
```

---

## Advantages

### 1. **Rapid Development**
- **Time Savings:** Go from idea to production in hours, not weeks
- **Pre-built Components:** No need to reinvent the wheel
- **Standardized Structure:** Consistent across projects

### 2. **Production-Ready Out-of-the-Box**
- **Docker Support:** One-command deployment
- **API Server:** REST API included with FastAPI
- **Monitoring:** Prometheus metrics and Grafana dashboards
- **CI/CD:** GitHub Actions workflows pre-configured

### 3. **Flexibility & Extensibility**
- **Multiple Data Sources:** CSV, databases, cloud storage
- **Pluggable Models:** Easy to add custom models
- **Configurable:** JSON/YAML configuration for everything
- **Modular Design:** Use only what you need

### 4. **MLOps Best Practices**
- **Experiment Tracking:** MLflow integration
- **Model Versioning:** Built-in model registry
- **Data Drift Detection:** Evidently.ai integration
- **Automated Testing:** Comprehensive test suite

### 5. **Enterprise-Grade Features**
- **Scalability:** Designed for production workloads
- **Security:** Input validation, environment variables
- **Observability:** Logging, metrics, monitoring
- **Documentation:** Auto-generated API docs

### 6. **Learning & Collaboration**
- **Well-Documented:** Extensive guides and examples
- **Type Hints:** Full type annotation for better IDE support
- **Code Quality:** Pre-commit hooks, linting, formatting
- **Testing:** Example tests to guide your own

### 7. **Cost-Effective**
- **Open Source:** Free to use and modify
- **No Vendor Lock-in:** Use any cloud provider
- **Resource Efficient:** Optimized for performance

---

## Current Limitations

### 1. **Model Support**

**Current Limitations:**
- Limited to scikit-learn compatible models
- No native deep learning support (TensorFlow, PyTorch)
- No GPU acceleration for training (except XGBoost)

**Workarounds:**
- You can extend the framework to add deep learning models
- Use the BaseModel interface to wrap any model type
- GPU support can be added through custom implementations

**Planned Improvements:**
- TensorFlow/Keras integration (v0.2.0)
- PyTorch support (v0.3.0)
- Automated GPU detection and usage

---

### 2. **Data Processing**

**Current Limitations:**
- In-memory processing only (not suitable for datasets > RAM)
- Limited streaming data support
- No automatic feature selection

**Workarounds:**
- Use chunked processing for large files
- Pre-process data externally (Spark, Dask)
- Implement custom feature selection

**Planned Improvements:**
- Dask integration for out-of-memory processing (v0.2.0)
- Streaming data connectors (Kafka, Kinesis) (v0.3.0)
- Automated feature engineering (v0.4.0)

---

### 3. **Deployment Options**

**Current Limitations:**
- Basic Kubernetes support (no Helm charts)
- No serverless deployment templates (AWS Lambda, Azure Functions)
- Limited edge deployment support

**Workarounds:**
- Create custom Kubernetes manifests
- Use Docker for containerization
- Export models to ONNX for edge deployment

**Planned Improvements:**
- Kubernetes Helm charts (v0.2.0)
- AWS SageMaker integration (v0.3.0)
- Serverless deployment templates (v0.3.0)

---

### 4. **Model Interpretability**

**Current Limitations:**
- Basic feature importance only
- No SHAP/LIME integration
- Limited model explanation tools

**Workarounds:**
- Use external libraries (SHAP, LIME) with exported models
- Implement custom explanation methods

**Planned Improvements:**
- SHAP integration (v0.2.0)
- Model explainability dashboard (v0.3.0)

---

### 5. **AutoML Capabilities**

**Current Limitations:**
- No automated model selection
- Hyperparameter tuning requires manual setup
- No neural architecture search

**Workarounds:**
- Use provided hyperparameter tuning with Optuna
- Manually compare multiple models

**Planned Improvements:**
- AutoML integration (Auto-sklearn, FLAML) (v0.3.0)
- Automated model selection pipeline (v0.4.0)

---

### 6. **Real-Time Features**

**Current Limitations:**
- No real-time feature store
- Limited online learning support
- Batch-oriented design

**Workarounds:**
- Use external feature stores (Feast)
- Implement custom online learning

**Planned Improvements:**
- Feature store integration (v0.3.0)
- Online learning support (v0.4.0)

---

### 7. **Multi-Model Support**

**Current Limitations:**
- One model per API instance
- No A/B testing built-in
- No model ensembling utilities

**Workarounds:**
- Deploy multiple instances for A/B testing
- Manually implement ensemble models

**Planned Improvements:**
- Multi-model serving (v0.2.0)
- A/B testing framework (v0.3.0)
- Ensemble utilities (v0.2.0)

---

## Roadmap

### Version 0.2.0 (Q2 2026)
- [ ] TensorFlow/Keras model support
- [ ] Dask integration for big data
- [ ] SHAP interpretability
- [ ] Kubernetes Helm charts
- [ ] Multi-model serving
- [ ] Model ensembling utilities

### Version 0.3.0 (Q3 2026)
- [ ] PyTorch support
- [ ] AWS SageMaker integration
- [ ] Streaming data connectors (Kafka)
- [ ] AutoML integration
- [ ] Feature store integration
- [ ] A/B testing framework

### Version 0.4.0 (Q4 2026)
- [ ] Automated feature engineering
- [ ] Online learning support
- [ ] Edge deployment templates
- [ ] Model governance dashboard
- [ ] Advanced AutoML capabilities

---

## Troubleshooting

### Common Issues

#### Issue 1: Import Errors After Installation

**Problem:**
```
ImportError: No module named 'ml_service'
```

**Solution:**
```bash
# Ensure you're in the correct virtual environment
which python  # Should show venv path

# Reinstall the package
pip uninstall ml-service-framework
pip install ml-service-framework
```

---

#### Issue 2: Model Training Fails with Memory Error

**Problem:**
```
MemoryError: Unable to allocate array
```

**Solution:**
```python
# Use chunked processing
processor = DataProcessor()
for chunk in pd.read_csv("large_file.csv", chunksize=10000):
    # Process in chunks
    pass

# Or reduce data size
df = df.sample(frac=0.5, random_state=42)  # Use 50% of data
```

---

#### Issue 3: Database Connection Fails

**Problem:**
```
Error: Could not connect to database
```

**Solution:**
```bash
# Check database is running
# PostgreSQL:
pg_isready -h localhost -p 5432

# MongoDB:
mongosh --eval "db.adminCommand('ping')"

# Verify credentials in .env file
cat .env | grep DB_
```

---

#### Issue 4: API Server Won't Start

**Problem:**
```
Error: Address already in use
```

**Solution:**
```bash
# Find process using port 8000
# Windows:
netstat -ano | findstr :8000

# Linux/Mac:
lsof -i :8000

# Kill the process or use different port
ml-serve --port 8001
```

---

#### Issue 5: Predictions Are Incorrect

**Checklist:**
- [ ] Ensure test data has same features as training data
- [ ] Check data preprocessing is applied consistently
- [ ] Verify model was loaded correctly
- [ ] Check for data drift
- [ ] Validate input data types

```python
# Debug predictions
service = InferenceService("models/model.joblib")

# Check what the model expects
print(service.data_processor.feature_columns)

# Validate input
test_input = pd.DataFrame([{"feature1": 1.0, "feature2": 2.0}])
prediction = service.predict(test_input)
print(f"Prediction: {prediction}")
```

---

## Additional Resources

### Documentation
- [README.md](README.md) - Main documentation
- [QUICKSTART.md](QUICKSTART.md) - 5-minute tutorial
- [CONTRIBUTING.md](CONTRIBUTING.md) - How to contribute
- [PUBLISHING.md](PUBLISHING.md) - Publishing guide

### Examples
- `docs/examples/` - Example use cases
- `config/` - Example configurations
- `tests/` - Example tests

### Community
- GitHub Issues: Report bugs and request features
- GitHub Discussions: Ask questions and share ideas
- Stack Overflow: Tag with `ml-service-framework`

### External Resources
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Evidently AI Documentation](https://docs.evidentlyai.com/)

---

## Support

If you encounter issues or have questions:

1. **Check Documentation:** Review this guide and the README
2. **Search Issues:** Look for similar issues on GitHub
3. **Create Issue:** Open a new issue with:
   - Clear description
   - Steps to reproduce
   - Expected vs actual behavior
   - System information (OS, Python version)
   - Error messages and logs

4. **Ask Community:** Post in GitHub Discussions

---

## License

This framework is released under the MIT License. See [LICENSE](LICENSE) file for details.

---

**Last Updated:** February 1, 2026  
**Version:** 0.1.0  
**Maintainers:** ML Service Framework Team
