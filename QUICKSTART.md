# Quick Start Guide

Get started with ML Service Framework in 5 minutes!

## Installation

```bash
pip install ml-service-framework
```

**Note:** The GitHub repo is MLOps-Boilerplate, but install via `ml-service-framework`.

## Create Your First Project

```bash
# Create a new project
ml-create-project my-first-ml-project

# Navigate to project
cd my-first-ml-project

# Set up environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Configure Your Project

```bash
# Copy environment template
cp .env.example .env

# Edit configuration
nano .env  # or use your favorite editor
```

## Train Your First Model

### 1. Prepare Your Data

Place your CSV data in `data/raw/train.csv` with a target column.

### 2. Update Configuration

Edit `config/training_config.json`:

```json
{
  "data_source": {
    "type": "file",
    "path": "data/raw/train.csv"
  },
  "target_column": "your_target_column",
  "model": {
    "type": "random_forest",
    "n_estimators": 100
  }
}
```

### 3. Train the Model

```bash
ml-train --config config/training_config.json
```

### 4. Run Inference

```bash
ml-inference --model-path models/model.joblib \
             --input-path data/raw/test.csv \
             --output-path predictions.csv
```

### 5. Serve via API

```bash
ml-serve --model-path models/model.joblib --port 8000
```

Test the API:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": {"feature1": 1.0, "feature2": 2.0}}'
```

## Next Steps

- 📚 Read the [full documentation](README.md)
- 🔧 Explore [configuration options](config/)
- 🧪 Check out [examples](docs/examples/)
- 🐳 Try [Docker deployment](docker-compose.yml)
- 📊 Set up [MLflow tracking](ml_service/machine_learning/experiment_tracking.py)

## Common Use Cases

### Train with Different Models

```python
# XGBoost
{
  "model": {
    "type": "xgboost",
    "n_estimators": 200,
    "learning_rate": 0.1
  }
}
```

### Use Database as Data Source

```json
{
  "data_source": {
    "type": "database",
    "connector_type": "postgresql",
    "connection_config": {
      "host": "localhost",
      "database": "mydb"
    },
    "query": "SELECT * FROM training_data"
  }
}
```

### Enable Experiment Tracking

```python
from ml_service.machine_learning.experiment_tracking import MLflowTracker

tracker = MLflowTracker()
tracker.log_training_run(config, metrics, model)
```

## Need Help?

- 💬 [Open an issue](https://github.com/kython220282/MLOps-Boilerplate/issues)
- 📖 [Read the docs](README.md)
- 🤝 [Contributing guide](CONTRIBUTING.md)
