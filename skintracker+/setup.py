#!/usr/bin/env python3
"""
Setup script for SkinTrack+
"""

import subprocess
import sys
import os
from pathlib import Path

def install_requirements():
    """Install required packages from requirements.txt"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All packages installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        return False
    return True

def create_directories():
    """Create necessary directories"""
    print("Creating data directories...")
    directories = ["skintrack_data", "skintrack_data/images", "skintrack_data/models"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    print("✅ Directories created successfully!")

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required!")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True

def main():
    print("🧴 SkinTrack+ Setup")
    print("=" * 50)
    
    if not check_python_version():
        sys.exit(1)
    
    create_directories()
    
    if install_requirements():
        print("\n🎉 Setup completed successfully!")
        print("\nTo run the application:")
        print("  streamlit run skintrack_app.py")
        print("\nFor help, see README.md")
    else:
        print("\n❌ Setup failed. Please check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
