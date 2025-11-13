"""
Setup Fabric Environment

This script validates that your environment is ready for the Delivery Time Prediction POC.
Run this before starting the deployment process.
"""

import sys
import subprocess
from pathlib import Path


def check_python_version():
    """Check if Python version is 3.10 or higher"""
    print("Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 10:
        print(f"✓ Python {version.major}.{version.minor}.{version.micro} detected")
        return True
    else:
        print(f"✗ Python 3.10+ required, but {version.major}.{version.minor} detected")
        return False


def check_required_packages():
    """Check if required packages are installed"""
    print("\nChecking required packages...")
    
    required_packages = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'sklearn': 'scikit-learn',
        'mlflow': 'mlflow',
        'flaml': 'FLAML',
        'sempy': 'semantic-link',
    }
    
    missing = []
    installed = []
    
    for import_name, package_name in required_packages.items():
        try:
            __import__(import_name)
            installed.append(package_name)
            print(f"✓ {package_name} installed")
        except ImportError:
            missing.append(package_name)
            print(f"✗ {package_name} NOT installed")
    
    return missing


def check_project_structure():
    """Verify project directory structure"""
    print("\nChecking project structure...")
    
    required_dirs = [
        'notebooks',
        'notebooks/utils',
        'config',
        'data/schema',
        'ml',
        'powerbi/dax',
        'scripts'
    ]
    
    project_root = Path(__file__).parent.parent
    missing_dirs = []
    
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if full_path.exists():
            print(f"✓ {dir_path}/ exists")
        else:
            print(f"✗ {dir_path}/ NOT found")
            missing_dirs.append(dir_path)
    
    return missing_dirs


def check_config_files():
    """Check if configuration files exist"""
    print("\nChecking configuration files...")
    
    required_files = [
        'config/environment.yml',
        'config/automl_settings.json',
        'config/fabric_lakehouse_paths.yaml',
    ]
    
    project_root = Path(__file__).parent.parent
    missing_files = []
    
    for file_path in required_files:
        full_path = project_root / file_path
        if full_path.exists():
            print(f"✓ {file_path} exists")
        else:
            print(f"✗ {file_path} NOT found")
            missing_files.append(file_path)
    
    return missing_files


def check_notebooks():
    """Check if required notebooks exist"""
    print("\nChecking notebooks...")
    
    required_notebooks = [
        '01_semantic_link_data_preparation.ipynb',
        '02_autoML_training_pipeline.ipynb',
        '03_batch_scoring_pipeline.ipynb',
    ]
    
    project_root = Path(__file__).parent.parent
    missing_notebooks = []
    
    for notebook in required_notebooks:
        full_path = project_root / notebook
        if full_path.exists():
            print(f"✓ {notebook} exists")
        else:
            print(f"✗ {notebook} NOT found")
            missing_notebooks.append(notebook)
    
    return missing_notebooks


def main():
    """Run all validation checks"""
    print("=" * 60)
    print("Delivery Time Prediction POC - Environment Setup Validation")
    print("=" * 60)
    
    checks_passed = True
    
    # Check Python version
    if not check_python_version():
        checks_passed = False
    
    # Check required packages
    missing_packages = check_required_packages()
    if missing_packages:
        checks_passed = False
        print(f"\n⚠ Install missing packages with:")
        print(f"   pip install {' '.join(missing_packages)}")
        print(f"   OR use: conda env create -f config/environment.yml")
    
    # Check project structure
    missing_dirs = check_project_structure()
    if missing_dirs:
        checks_passed = False
    
    # Check config files
    missing_files = check_config_files()
    if missing_files:
        checks_passed = False
    
    # Check notebooks
    missing_notebooks = check_notebooks()
    if missing_notebooks:
        checks_passed = False
    
    # Summary
    print("\n" + "=" * 60)
    if checks_passed:
        print("✓ All checks passed! Environment is ready.")
        print("\nNext steps:")
        print("1. Update config/fabric_lakehouse_paths.yaml with your Fabric IDs")
        print("2. Run: python scripts/validate_semantic_model.py")
        print("3. Upload notebooks to Fabric workspace")
    else:
        print("✗ Some checks failed. Please fix the issues above.")
        return 1
    
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
