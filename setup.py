"""Setup configuration for ML Service Framework."""
from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8') if (this_directory / "README.md").exists() else ""

# Read requirements
requirements = []
if (this_directory / "requirements.txt").exists():
    with open(this_directory / "requirements.txt") as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="ml-service-framework",
    version="0.1.0",
    author="Karan",
    author_email="kython220282@gmail.com",
    description="A production-ready MLOps framework for building, training, deploying, and monitoring ML models at scale",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kython220282/MLOps-Boilerplate",
    packages=find_packages(exclude=["tests", "tests.*", "docs"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.10.0",
            "flake8>=6.1.0",
            "mypy>=1.6.0",
            "pre-commit>=3.5.0",
        ],
        "docs": [
            "sphinx>=7.2.0",
            "sphinx-rtd-theme>=1.3.0",
            "mkdocs>=1.5.0",
            "mkdocs-material>=9.4.0",
        ],
        "monitoring": [
            "prometheus-client>=0.18.0",
            "evidently>=0.4.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "ml-train=ml_service.applications.training:main",
            "ml-inference=ml_service.applications.inference:main",
            "ml-serve=ml_service.applications.api_server:main",
            "ml-create-project=ml_service.cli.create_project:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
