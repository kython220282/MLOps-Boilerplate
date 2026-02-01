# Contributing to ML Service Framework

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing.

## 🌟 How to Contribute

### Reporting Bugs

1. Check if the bug has already been reported in [Issues](https://github.com/yourusername/ml-service-framework/issues)
2. If not, create a new issue with:
   - Clear, descriptive title
   - Steps to reproduce
   - Expected vs actual behavior
   - Environment details (OS, Python version, etc.)
   - Code samples or error messages

### Suggesting Features

1. Check existing feature requests
2. Create a new issue with tag `enhancement`
3. Describe the feature and its use case
4. Explain why it would be useful

### Pull Requests

1. **Fork the repository**
2. **Create a branch** from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**:
   - Follow the code style guidelines
   - Add tests for new functionality
   - Update documentation

4. **Test your changes**:
   ```bash
   pytest
   black ml_service tests
   flake8 ml_service tests
   mypy ml_service
   ```

5. **Commit with clear messages**:
   ```bash
   git commit -m "Add feature: description of feature"
   ```

6. **Push and create PR**:
   ```bash
   git push origin feature/your-feature-name
   ```

## 📝 Code Style Guidelines

### Python Style

- Follow PEP 8
- Use Black for formatting (line length: 100)
- Use type hints where possible
- Write descriptive docstrings

### Example:

```python
def process_data(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Process DataFrame by transforming specified column.
    
    Args:
        df: Input DataFrame
        column: Name of column to process
    
    Returns:
        Processed DataFrame
    
    Raises:
        ValueError: If column doesn't exist
    """
    if column not in df.columns:
        raise ValueError(f"Column {column} not found")
    
    # Processing logic here
    return df
```

### Documentation

- Add docstrings to all public functions/classes
- Update README.md for new features
- Add examples in `docs/examples/`

### Testing

- Write unit tests for all new code
- Aim for >80% code coverage
- Use descriptive test names
- Test edge cases and error conditions

```python
def test_process_data_with_missing_column():
    """Test that ValueError is raised for missing column."""
    df = pd.DataFrame({'a': [1, 2, 3]})
    with pytest.raises(ValueError):
        process_data(df, 'missing_column')
```

## 🔄 Development Workflow

1. **Set up development environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   pip install -e ".[dev]"
   ```

2. **Install pre-commit hooks**:
   ```bash
   pre-commit install
   ```

3. **Make changes and test**:
   ```bash
   # Run tests
   pytest
   
   # Check formatting
   black --check ml_service tests
   
   # Run linting
   flake8 ml_service tests
   
   # Type checking
   mypy ml_service
   ```

4. **Build documentation** (if applicable):
   ```bash
   cd docs
   make html
   ```

## 🏗️ Project Structure

When adding new features, maintain the existing structure:

```
ml_service/
├── applications/      # User-facing applications
├── data_layer/       # Data access and storage
├── machine_learning/ # Core ML functionality
└── config.py        # Configuration management
```

## ✅ Checklist Before Submitting PR

- [ ] Code follows style guidelines
- [ ] Tests added and passing
- [ ] Documentation updated
- [ ] No merge conflicts with main
- [ ] Commits are clear and descriptive
- [ ] PR description explains changes

## 🎯 Good First Issues

Look for issues labeled `good first issue` - these are great starting points for new contributors!

## 💬 Questions?

Feel free to:
- Open a discussion on GitHub
- Ask in pull request comments
- Reach out to maintainers

Thank you for contributing! 🎉
