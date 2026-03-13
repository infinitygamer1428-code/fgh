# setup.py
import subprocess
import sys
import os

def install_packages():
    packages = [
        'Flask==2.3.3',
        'Flask-SQLAlchemy==3.0.5',
        'Flask-Login==0.6.2',
        'opencv-python==4.8.1.78',
        'face-recognition==1.3.0',
        'numpy==1.24.3',
        'Pillow==10.0.0',
        'Werkzeug==2.3.7'
    ]
    
    for package in packages:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    print("\nAll packages installed successfully!")

if __name__ == "__main__":
    install_packages()