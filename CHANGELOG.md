# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial release of ML Service Framework

## [0.1.0] - 2026-02-01

### Added
- Core ML pipeline with training and inference
- Data layer with PostgreSQL, MongoDB, S3, and Azure Blob connectors
- Multiple ML models: Random Forest, XGBoost
- Data preprocessing pipeline with feature engineering
- Cross-validation and model evaluation
- Hyperparameter tuning (Grid Search, Random Search, Optuna)
- MLflow experiment tracking integration
- Model registry for version management
- FastAPI REST API for model serving
- Prometheus metrics and monitoring
- Data drift detection using Evidently
- Docker and docker-compose support
- Comprehensive test suite with pytest
- CI/CD with GitHub Actions
- Pre-commit hooks for code quality
- Pydantic-based configuration management
- CLI tool for creating new projects (`ml-create-project`)
- Complete documentation and examples

### Security
- Added security scanning in CI/CD pipeline
- Input validation with Pydantic

[Unreleased]: https://github.com/kython220282/MLOps-Boilerplate/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/kython220282/MLOps-Boilerplate/releases/tag/v0.1.0
