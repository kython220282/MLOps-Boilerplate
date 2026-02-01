"""CLI tool to create new ML projects from template."""
import os
import sys
import shutil
import argparse
from pathlib import Path
from typing import Optional
import importlib.resources as pkg_resources


def create_project(project_name: str, output_dir: Optional[str] = None) -> None:
    """
    Create a new ML project from the template.
    
    Args:
        project_name: Name of the new project
        output_dir: Output directory (defaults to current directory)
    """
    # Determine output directory
    if output_dir is None:
        output_dir = os.getcwd()
    
    project_path = Path(output_dir) / project_name
    
    if project_path.exists():
        print(f"Error: Directory '{project_path}' already exists!")
        sys.exit(1)
    
    print(f"Creating ML project '{project_name}' in {output_dir}...")
    
    # Create project structure
    create_directory_structure(project_path, project_name)
    
    # Copy template files
    copy_template_files(project_path, project_name)
    
    print(f"\n✅ Project '{project_name}' created successfully!")
    print(f"\n📁 Project location: {project_path}")
    print("\n🚀 Next steps:")
    print(f"   cd {project_name}")
    print("   python -m venv venv")
    print("   source venv/bin/activate  # On Windows: venv\\Scripts\\activate")
    print("   pip install -r requirements.txt")
    print("   cp .env.example .env")
    print("   # Edit .env with your configuration")
    print("   ml-train --config config/training_config.json")
    print("\n📚 Read README.md for detailed documentation")


def create_directory_structure(project_path: Path, project_name: str) -> None:
    """Create the project directory structure."""
    
    # Main package directory
    package_name = project_name.replace('-', '_').lower()
    
    directories = [
        f"{package_name}",
        f"{package_name}/applications",
        f"{package_name}/data_layer",
        f"{package_name}/machine_learning",
        f"{package_name}/monitoring",
        "config",
        "tests",
        "docs",
        "docs/examples",
        "models",
        "logs",
        "data",
        "data/raw",
        "data/processed",
        "monitoring",
        ".github/workflows",
    ]
    
    for directory in directories:
        (project_path / directory).mkdir(parents=True, exist_ok=True)
        print(f"  Created: {directory}/")


def copy_template_files(project_path: Path, project_name: str) -> None:
    """Copy template files to the new project."""
    
    package_name = project_name.replace('-', '_').lower()
    
    # Get the template directory from the installed package
    try:
        import ml_service
        template_dir = Path(ml_service.__file__).parent.parent
    except:
        print("Warning: Could not locate template files. Creating minimal structure.")
        create_minimal_files(project_path, package_name)
        return
    
    # Files to copy with their destinations
    files_to_copy = {
        'requirements.txt': 'requirements.txt',
        'setup.py': 'setup.py',
        '.env.example': '.env.example',
        'pyproject.toml': 'pyproject.toml',
        '.gitignore': '.gitignore',
        '.flake8': '.flake8',
        '.pre-commit-config.yaml': '.pre-commit-config.yaml',
        'README.md': 'README.md',
        'CONTRIBUTING.md': 'CONTRIBUTING.md',
        'LICENSE': 'LICENSE',
        'Dockerfile': 'Dockerfile',
        'docker-compose.yml': 'docker-compose.yml',
        '.dockerignore': '.dockerignore',
    }
    
    # Copy files
    for src, dst in files_to_copy.items():
        src_path = template_dir / src
        dst_path = project_path / dst
        if src_path.exists():
            shutil.copy2(src_path, dst_path)
            print(f"  Created: {dst}")
    
    # Copy Python package files
    copy_package_files(template_dir / 'ml_service', project_path / package_name, package_name)
    
    # Copy config files
    copy_directory(template_dir / 'config', project_path / 'config')
    
    # Copy test files
    copy_directory(template_dir / 'tests', project_path / 'tests')
    
    # Copy monitoring files
    copy_directory(template_dir / 'monitoring', project_path / 'monitoring')
    
    # Copy GitHub workflows
    copy_directory(template_dir / '.github', project_path / '.github')
    
    # Update package name in files
    update_package_name(project_path, 'ml_service', package_name, project_name)


def copy_package_files(src_dir: Path, dst_dir: Path, package_name: str) -> None:
    """Copy Python package files."""
    if not src_dir.exists():
        return
    
    for item in src_dir.rglob('*.py'):
        if '__pycache__' in str(item):
            continue
        
        relative_path = item.relative_to(src_dir)
        dst_path = dst_dir / relative_path
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, dst_path)


def copy_directory(src_dir: Path, dst_dir: Path) -> None:
    """Copy entire directory."""
    if not src_dir.exists():
        return
    
    for item in src_dir.rglob('*'):
        if '__pycache__' in str(item) or item.is_dir():
            continue
        
        relative_path = item.relative_to(src_dir)
        dst_path = dst_dir / relative_path
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, dst_path)


def update_package_name(project_path: Path, old_name: str, new_name: str, project_name: str) -> None:
    """Update package name in all files."""
    
    files_to_update = [
        'setup.py',
        'README.md',
        f'{new_name}/__init__.py',
    ]
    
    for file_name in files_to_update:
        file_path = project_path / file_name
        if file_path.exists():
            try:
                content = file_path.read_text()
                content = content.replace(old_name, new_name)
                content = content.replace('ml-service-framework', project_name)
                content = content.replace('ML Service Framework', project_name.replace('-', ' ').title())
                file_path.write_text(content)
            except Exception as e:
                print(f"  Warning: Could not update {file_name}: {e}")


def create_minimal_files(project_path: Path, package_name: str) -> None:
    """Create minimal template files when templates aren't available."""
    
    # Create basic __init__.py files
    init_content = f'"""{package_name.replace("_", " ").title()} Package."""\n\n__version__ = "0.1.0"\n'
    
    for subdir in ['', 'applications', 'data_layer', 'machine_learning', 'monitoring']:
        init_path = project_path / package_name / subdir / '__init__.py'
        init_path.write_text(init_content)
    
    # Create basic README
    readme_content = f"""# {package_name.replace('_', ' ').title()}

A machine learning project created with ML Service Framework.

## Setup

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate
pip install -r requirements.txt
```

## Usage

1. Configure your project in `.env`
2. Add your data to `data/raw/`
3. Update configuration in `config/`
4. Train your model: `ml-train --config config/training_config.json`
5. Run inference: `ml-inference --model-path models/model.joblib --input-path data/test.csv`

## License

MIT License
"""
    (project_path / 'README.md').write_text(readme_content)
    
    print("  Created minimal project structure")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Create a new ML project from the ML Service Framework template',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  ml-create-project my-ml-project
  ml-create-project my-ml-project --output-dir /path/to/projects
  
This will create a complete ML project structure with:
  - Data layer with database and cloud storage connectors
  - ML components (models, preprocessing, training pipeline)
  - REST API for model serving
  - Monitoring and drift detection
  - Docker support
  - CI/CD configuration
  - Comprehensive tests
        """
    )
    
    parser.add_argument(
        'project_name',
        help='Name of the project to create (e.g., my-ml-project)'
    )
    
    parser.add_argument(
        '--output-dir',
        '-o',
        default=None,
        help='Output directory (default: current directory)'
    )
    
    args = parser.parse_args()
    
    # Validate project name
    if not args.project_name.replace('-', '').replace('_', '').isalnum():
        print("Error: Project name should contain only letters, numbers, hyphens, and underscores")
        sys.exit(1)
    
    create_project(args.project_name, args.output_dir)


if __name__ == '__main__':
    main()
