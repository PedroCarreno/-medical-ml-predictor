#!/usr/bin/env python3
"""
🚀 Medical ML Predictor - Project Setup Script
============================================

Automated setup script for the complete medical ML prediction system.
Handles environment setup, dependencies, and initial configuration.

Usage:
    python scripts/setup_project.py
"""

import os
import sys
import subprocess
import platform
from pathlib import Path
import urllib.request
import json

class MedicalMLSetup:
    """🏥 Automated setup for Medical ML Predictor project"""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.system = platform.system().lower()
        self.python_version = f"{sys.version_info.major}.{sys.version_info.minor}"

    def print_header(self):
        """Print setup header"""
        print("🏥" + "="*60 + "🏥")
        print("   MEDICAL ML PREDICTOR - AUTOMATED SETUP")
        print("   🎯 Hospital Mortality Prediction System")
        print("="*64)
        print(f"🖥️  System: {platform.system()} {platform.release()}")
        print(f"🐍 Python: {sys.version}")
        print(f"📁 Project: {self.project_root}")
        print("="*64)

    def check_python_version(self):
        """Check if Python version is compatible"""
        print("\n🐍 Checking Python version...")

        if sys.version_info < (3, 8):
            print("❌ Python 3.8+ required. Current version:", self.python_version)
            print("   Please upgrade Python and try again.")
            return False

        print(f"✅ Python {self.python_version} is compatible")
        return True

    def check_dependencies(self):
        """Check if required system dependencies are available"""
        print("\n🔧 Checking system dependencies...")

        dependencies = ['git', 'docker', 'docker-compose']
        missing = []

        for dep in dependencies:
            try:
                subprocess.run([dep, '--version'],
                             capture_output=True, check=True)
                print(f"✅ {dep} is available")
            except (subprocess.CalledProcessError, FileNotFoundError):
                print(f"⚠️  {dep} not found (optional for core functionality)")
                missing.append(dep)

        if missing:
            print(f"\n📝 Optional dependencies missing: {', '.join(missing)}")
            print("   Core ML functionality will still work!")

        return True

    def create_virtual_environment(self):
        """Create and activate virtual environment"""
        print("\n🌿 Setting up virtual environment...")

        venv_path = self.project_root / '.venv'

        if venv_path.exists():
            print("✅ Virtual environment already exists")
            return True

        try:
            subprocess.run([sys.executable, '-m', 'venv', str(venv_path)],
                         check=True)
            print("✅ Virtual environment created successfully")

            # Instructions for activation
            if self.system == 'windows':
                activate_cmd = str(venv_path / 'Scripts' / 'activate.bat')
            else:
                activate_cmd = f"source {venv_path}/bin/activate"

            print(f"📝 To activate: {activate_cmd}")
            return True

        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to create virtual environment: {e}")
            return False

    def install_requirements(self):
        """Install Python requirements"""
        print("\n📦 Installing Python dependencies...")

        requirements_file = self.project_root / 'requirements.txt'

        if not requirements_file.exists():
            print("❌ requirements.txt not found")
            return False

        try:
            # Try to install with current Python
            subprocess.run([
                sys.executable, '-m', 'pip', 'install', '-r',
                str(requirements_file)
            ], check=True)

            print("✅ Dependencies installed successfully")
            return True

        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install dependencies: {e}")
            print("💡 Try running: pip install -r requirements.txt")
            return False

    def create_directories(self):
        """Create necessary project directories"""
        print("\n📁 Creating project directories...")

        directories = [
            'data/processed',
            'data/raw',
            'ml_models/saved_models',
            'ml_models/results',
            'ml_models/plots',
            'notebooks/outputs',
            'logs',
            'tests/unit',
            'tests/integration',
            'api_docs',
            'scripts/deployment'
        ]

        for dir_path in directories:
            full_path = self.project_root / dir_path
            full_path.mkdir(parents=True, exist_ok=True)

        print(f"✅ Created {len(directories)} directories")
        return True

    def setup_environment_file(self):
        """Setup .env file if it doesn't exist"""
        print("\n🔧 Setting up environment configuration...")

        env_file = self.project_root / '.env'

        if env_file.exists():
            print("✅ .env file already exists")
            return True

        env_content = '''# 🏥 Medical ML Predictor Environment Configuration

# Flask Configuration
FLASK_ENV=development
FLASK_SECRET_KEY=your-secret-key-here-change-in-production
FLASK_HOST=0.0.0.0
FLASK_PORT=8000

# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=medical_predictions
DB_USER=medml_user
DB_PASSWORD=secure_password_change_me

# ML Model Configuration
MODEL_PATH=ml_models/saved_models/
DEFAULT_MODEL=best_model.joblib
PREDICTION_THRESHOLD=0.5

# API Configuration
API_VERSION=v1
MAX_REQUESTS_PER_MINUTE=100

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/medical_ml.log

# Security
JWT_SECRET_KEY=your-jwt-secret-here
SESSION_TIMEOUT=3600

# Data Configuration
DATASET_PATH=dataset.csv
BACKUP_PATH=data/backups/

# Feature Flags
ENABLE_MONITORING=true
ENABLE_CACHING=true
ENABLE_MODEL_EXPLANATIONS=true

# Performance
MAX_BATCH_SIZE=1000
TIMEOUT_SECONDS=30
'''

        try:
            with open(env_file, 'w') as f:
                f.write(env_content)
            print("✅ .env file created")
            print("⚠️  Remember to update passwords and secrets!")
            return True
        except Exception as e:
            print(f"❌ Failed to create .env file: {e}")
            return False

    def verify_dataset(self):
        """Check if dataset is available"""
        print("\n📊 Checking dataset availability...")

        dataset_path = self.project_root / 'dataset.csv'

        if dataset_path.exists():
            # Check file size
            size_mb = dataset_path.stat().st_size / (1024 * 1024)
            print(f"✅ Dataset found: {size_mb:.1f} MB")

            # Quick validation
            try:
                import pandas as pd
                df = pd.read_csv(dataset_path, nrows=5)
                print(f"✅ Dataset validated: {len(df.columns)} columns")
                return True
            except Exception as e:
                print(f"⚠️  Dataset found but validation failed: {e}")
                return True  # File exists, might just be import issue
        else:
            print("⚠️  Dataset not found at: dataset.csv")
            print("📝 Please ensure your dataset.csv is in the project root")
            print("   Expected format: 91,713+ rows with 84+ columns")
            return False

    def create_quick_start_script(self):
        """Create a quick start script"""
        print("\n🚀 Creating quick start script...")

        if self.system == 'windows':
            script_name = 'quick_start.bat'
            script_content = '''@echo off
echo 🏥 Medical ML Predictor - Quick Start
echo ====================================

echo 🐍 Activating virtual environment...
call .venv\\Scripts\\activate.bat

echo 📊 Starting Jupyter for EDA...
echo Open: http://localhost:8888
jupyter notebook notebooks/exploratory_data_analysis.ipynb

pause
'''
        else:
            script_name = 'quick_start.sh'
            script_content = '''#!/bin/bash
echo "🏥 Medical ML Predictor - Quick Start"
echo "===================================="

echo "🐍 Activating virtual environment..."
source .venv/bin/activate

echo "📊 Starting Jupyter for EDA..."
echo "Open: http://localhost:8888"
jupyter notebook notebooks/exploratory_data_analysis.ipynb
'''

        script_path = self.project_root / script_name

        try:
            with open(script_path, 'w') as f:
                f.write(script_content)

            if self.system != 'windows':
                os.chmod(script_path, 0o755)

            print(f"✅ Quick start script created: {script_name}")
            return True
        except Exception as e:
            print(f"❌ Failed to create quick start script: {e}")
            return False

    def print_next_steps(self):
        """Print next steps for the user"""
        print("\n🎉 SETUP COMPLETED SUCCESSFULLY!")
        print("="*50)
        print("\n📋 NEXT STEPS:")
        print("\n1. 📊 Explore the Data:")
        print("   • Open: notebooks/exploratory_data_analysis.ipynb")
        print("   • Run: jupyter notebook")

        print("\n2. 🤖 Train ML Models:")
        print("   • Run: python ml_models/train_models.py")
        print("   • Check results in: ml_models/results/")

        print("\n3. 🌐 Start Web Application:")
        print("   • Run: docker-compose up --build")
        print("   • Access: http://localhost:3000")

        print("\n4. 📈 View Model Performance:")
        print("   • Check: ml_models/plots/")
        print("   • Read: ml_models/results/model_performance_report.txt")

        print("\n🔧 USEFUL COMMANDS:")
        print("   📊 EDA: jupyter notebook notebooks/")
        print("   🤖 Train: python ml_models/train_models.py")
        print("   🌐 Web App: docker-compose up")
        print("   🧪 Tests: pytest tests/")

        print("\n📚 DOCUMENTATION:")
        print("   • README.md - Complete project guide")
        print("   • Dataset-Info.pdf - Data description")
        print("   • docs/ - Additional documentation")

        print("\n⚠️  IMPORTANT REMINDERS:")
        print("   • Update .env file with secure passwords")
        print("   • Ensure dataset.csv is in project root")
        print("   • Activate virtual environment before running")

        print(f"\n✨ Happy ML modeling! 🏥🤖")

    def run_setup(self):
        """Run the complete setup process"""
        self.print_header()

        steps = [
            ("🐍 Python Version", self.check_python_version),
            ("🔧 Dependencies", self.check_dependencies),
            ("🌿 Virtual Environment", self.create_virtual_environment),
            ("📦 Install Requirements", self.install_requirements),
            ("📁 Create Directories", self.create_directories),
            ("🔧 Environment File", self.setup_environment_file),
            ("📊 Dataset Check", self.verify_dataset),
            ("🚀 Quick Start Script", self.create_quick_start_script)
        ]

        failed_steps = []

        for step_name, step_func in steps:
            print(f"\n{'='*20}")
            try:
                success = step_func()
                if not success:
                    failed_steps.append(step_name)
            except Exception as e:
                print(f"❌ {step_name} failed with error: {e}")
                failed_steps.append(step_name)

        if failed_steps:
            print(f"\n⚠️  Some steps failed: {', '.join(failed_steps)}")
            print("   You may need to complete these manually.")

        self.print_next_steps()
        return len(failed_steps) == 0

def main():
    """Main setup function"""
    setup = MedicalMLSetup()
    success = setup.run_setup()

    if success:
        print("\n🎉 Setup completed successfully!")
        sys.exit(0)
    else:
        print("\n⚠️  Setup completed with some issues.")
        print("   Check the output above for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()