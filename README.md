# MLOps-Boilerplate

A production-ready machine learning framework for building, training, deploying, and monitoring ML models at scale.

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## 🚀 Features

- **Modular Architecture**: Clean separation of concerns with data layer, ML components, and applications
- **Multiple Data Sources**: Built-in connectors for PostgreSQL, MongoDB, AWS S3, Azure Blob Storage
- **ML Models**: Support for Random Forest, XGBoost, and easy extensibility for custom models
- **Data Processing**: Complete preprocessing pipeline with feature engineering and scaling
- **Cross-Validation**: K-Fold and Stratified K-Fold validation with comprehensive metrics
- **Experiment Tracking**: Integration with MLflow for experiment management
- **Configuration Management**: Pydantic-based configuration with validation
- **Testing**: Comprehensive test suite with pytest
- **Monitoring**: Model performance tracking and data drift detection
- **API Server**: FastAPI-based REST API for model serving
- **Production Ready**: Docker support, CI/CD pipelines, and best practices

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Usage Examples](#usage-examples)
- [Configuration](#configuration)
- [Testing](#testing)
- [API Documentation](#api-documentation)
- [Contributing](#contributing)
- [License](#license)

## 🔧 Installation

### Install from PyPI (Recommended)

```bash
pip install mlops-boilerplate
```

### Create a New Project

After installation, create a new ML project using the template:

```bash
# Create a new project
ml-create-project my-ml-project

# Navigate to your project
cd my-ml-project

# Set up virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your configuration
```

### Install from Source (For Development)

```bash
# Clone the repository
git clone https://github.com/kython220282/MLOps-Boilerplate.git
cd MLOps-Boilerplate

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# Install in development mode
pip install -e .
```

### Install with Docker

```bash
docker build -t mlops-boilerplate .
docker run -p 8000:8000 mlops-boilerplate
```

## 🏃 Quick Start

### 1. Setup Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your configuration
```

### 2. Train a Model

```bash
# Using CLI
ml-train --config config/training_config.json

# Or with Python
python -m ml_service.applications.training --config config/training_config.json
```

### 3. Run Inference

```bash
ml-inference --model-path models/model.joblib \\
             --input-path data/test.csv \\
             --output-path predictions.csv
```

### 4. Start API Server

```bash
ml-serve --model-path models/model.joblib --port 8000
```

## 📁 Project Structure

```
machine_learning_service/
├── ml_service/                 # Main package
│   ├── applications/          # Application entry points
│   │   ├── training.py       # Training CLI application
│   │   └── inference.py      # Inference CLI application
│   ├── data_layer/           # Data connectors
│   │   ├── data_connector.py # Database connectors
│   │   └── object_connector.py # Cloud storage connectors
│   ├── machine_learning/     # ML components
│   │   ├── data_processor.py # Data preprocessing
│   │   ├── model.py          # Model definitions
│   │   ├── training_pipeline.py # Training orchestration
│   │   └── cross_validator.py # Model validation
│   └── config.py             # Configuration management
├── config/                    # Configuration files
│   ├── training_config.json
│   └── training_config.yaml
├── tests/                     # Test suite
├── docs/                      # Documentation
├── requirements.txt          # Dependencies
├── setup.py                  # Package setup
├── pyproject.toml           # Project configuration
├── .env.example             # Environment template
└── README.md                # This file
```

## 💡 Usage Examples

### Training with Different Data Sources

**From CSV File:**
```python
from ml_service.machine_learning.training_pipeline import TrainingPipeline

config = {
    "data_source": {"type": "file", "path": "data/train.csv"},
    "target_column": "target",
    "model": {"type": "random_forest", "n_estimators": 100},
    "task_type": "classification"
}

pipeline = TrainingPipeline(config)
metrics = pipeline.run_pipeline()
```

**From Database:**
```python
config = {
    "data_source": {
        "type": "database",
        "connector_type": "postgresql",
        "connection_config": {
            "host": "localhost",
            "database": "ml_db"
        },
        "query": "SELECT * FROM training_data"
    },
    "target_column": "target"
}
```

### Custom Model Development

```python
from ml_service.machine_learning.model import BaseModel

class CustomModel(BaseModel):
    def build_model(self):
        # Your model architecture
        pass
    
    def train(self, X_train, y_train):
        # Training logic
        pass
    
    def predict(self, X):
        # Prediction logic
        pass
```

### Data Processing Pipeline

```python
from ml_service.machine_learning.data_processor import DataProcessor

processor = DataProcessor()
df = processor.load_data("data/train.csv")
X_train, X_test, y_train, y_test = processor.preprocess_pipeline(
    df, target_column="target"
)
```

## ⚙️ Configuration

### Training Configuration

Create a JSON or YAML configuration file:

```json
{
  "data_source": {
    "type": "file",
    "path": "data/train.csv"
  },
  "target_column": "target",
  "model": {
    "type": "random_forest",
    "n_estimators": 100,
    "max_depth": 10
  },
  "data_processing": {
    "missing_value_strategy": "mean",
    "scaling_method": "standard"
  },
  "cross_validation": {
    "type": "kfold",
    "n_splits": 5
  }
}
```

### Environment Variables

Set in `.env` file:

```bash
# Database
DB_TYPE=postgresql
DB_HOST=localhost
DB_PORT=5432

# MLflow
MLFLOW_TRACKING_URI=http://localhost:5000

# API
API_HOST=0.0.0.0
API_PORT=8000
```

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=ml_service --cov-report=html

# Run specific test file
pytest tests/test_model.py

# Run with markers
pytest -m unit
pytest -m integration
```

## 📚 API Documentation

Start the API server and visit `http://localhost:8000/docs` for interactive Swagger documentation.

### Example API Endpoints

**Health Check:**
```bash
curl http://localhost:8000/health
```

**Make Prediction:**
```bash
curl -X POST http://localhost:8000/predict \\
  -H "Content-Type: application/json" \\
  -d '{"features": [1.5, 2.3, 3.1, 4.2, 5.0]}'
```

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install dev dependencies
pip install -r requirements.txt
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run code formatting
black ml_service tests
isort ml_service tests

# Run linting
flake8 ml_service tests
mypy ml_service
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- scikit-learn for ML algorithms
- MLflow for experiment tracking
- FastAPI for API framework
- Pydantic for configuration management

## �‍💻 Credits

**Created by:** Karan  
**GitHub:** [@kython220282](https://github.com/kython220282)  
**Repository:** [MLOps-Boilerplate](https://github.com/kython220282/MLOps-Boilerplate)

### 🌟 If You Use This Framework

If you use this framework in your projects, please consider:
- ⭐ **Star this repository** on GitHub
- 📝 **Add credits** in your project documentation:
  ```markdown
  Built with [MLOps-Boilerplate](https://github.com/kython220282/MLOps-Boilerplate) by Karan
  ```
- 🔗 **Link back** to this repository
- 💬 **Share your project** - Open an issue to showcase what you've built!

Your support helps maintain and improve this framework for everyone. Thank you! 🙏

## 📞 Support

For questions and support:
- Create an issue on [GitHub](https://github.com/kython220282/MLOps-Boilerplate/issues)
- Email: support@example.com
- Repository: https://github.com/kython220282/MLOps-Boilerplate

---

**Happy Model Building! 🎉**
