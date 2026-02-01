# Publishing ML Service Framework to PyPI

This guide explains how to publish the ML Service Framework to PyPI so others can install it with `pip install ml-service-framework`.

## Prerequisites

1. **PyPI Account**
   - Create account at https://pypi.org/account/register/
   - (Optional) Create TestPyPI account at https://test.pypi.org/account/register/

2. **Install Build Tools**
   ```bash
   pip install --upgrade build twine
   ```

3. **Create API Token**
   - Go to https://pypi.org/manage/account/token/
   - Create a new API token
   - Save the token securely (you'll need it for uploading)

## Step-by-Step Publishing Process

### 1. Update Version Number

Edit `setup.py` and `ml_service/__init__.py` with the new version:

```python
version="0.1.0"  # Update this
```

### 2. Clean Previous Builds

```bash
# Remove old build artifacts
rm -rf build dist *.egg-info
```

### 3. Build the Package

```bash
# Build source distribution and wheel
python -m build
```

This creates files in the `dist/` directory:
- `ml-service-framework-0.1.0.tar.gz` (source distribution)
- `ml_service_framework-0.1.0-py3-none-any.whl` (wheel)

### 4. Test Upload to TestPyPI (Optional but Recommended)

```bash
# Upload to TestPyPI
python -m twine upload --repository testpypi dist/*

# Test installation from TestPyPI
pip install --index-url https://test.pypi.org/simple/ ml-service-framework
```

### 5. Upload to PyPI

```bash
# Upload to PyPI
python -m twine upload dist/*
```

When prompted:
- Username: `__token__`
- Password: Your PyPI API token (including the `pypi-` prefix)

### 6. Verify Installation

```bash
# Install from PyPI
pip install ml-service-framework

# Test the CLI
ml-create-project test-project
```

## Automated Publishing with GitHub Actions

Create `.github/workflows/publish.yml`:

```yaml
name: Publish to PyPI

on:
  release:
    types: [published]

jobs:
  deploy:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install build twine
    
    - name: Build package
      run: python -m build
    
    - name: Publish to PyPI
      env:
        TWINE_USERNAME: __token__
        TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
      run: python -m twine upload dist/*
```

**Setup GitHub Secret:**
1. Go to repository Settings → Secrets → Actions
2. Add new secret: `PYPI_API_TOKEN` with your PyPI token

## Version Management

Follow [Semantic Versioning](https://semver.org/):
- **Major** (1.0.0): Breaking changes
- **Minor** (0.1.0): New features, backward compatible
- **Patch** (0.0.1): Bug fixes

## Release Checklist

- [ ] Update version in `setup.py`
- [ ] Update version in `ml_service/__init__.py`
- [ ] Update CHANGELOG.md
- [ ] Update README.md if needed
- [ ] Run tests: `pytest`
- [ ] Build package: `python -m build`
- [ ] Test locally: `pip install dist/*.whl`
- [ ] Upload to TestPyPI (optional)
- [ ] Upload to PyPI
- [ ] Create GitHub release
- [ ] Update documentation

## Package Maintenance

### Update Package

1. Make changes to code
2. Update version number
3. Rebuild and republish

### Yanking a Release (if needed)

If you need to remove a broken release:

```bash
# Using twine
twine yank ml-service-framework <version>

# Or via PyPI web interface
# Go to: https://pypi.org/project/ml-service-framework/
```

## Troubleshooting

**Build errors:**
```bash
# Clean everything
rm -rf build dist *.egg-info
python -m build
```

**Upload errors:**
- Check your API token is correct
- Ensure version number is unique (you can't reupload same version)
- Check package name isn't already taken

**Import errors after install:**
- Ensure `__init__.py` files exist in all packages
- Check MANIFEST.in includes all necessary files

## References

- [PyPI Publishing Guide](https://packaging.python.org/tutorials/packaging-projects/)
- [Twine Documentation](https://twine.readthedocs.io/)
- [Semantic Versioning](https://semver.org/)
