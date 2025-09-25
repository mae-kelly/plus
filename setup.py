#!/usr/bin/env python3
"""
CMDB+ Setup Script
Verifies environment and dependencies before running
"""

import sys
import platform
import subprocess
import json
from pathlib import Path

def check_mac_silicon():
    """Check if running on Apple Silicon Mac"""
    if platform.system() != 'Darwin':
        return False
        
    try:
        result = subprocess.run(
            ['sysctl', '-n', 'hw.optional.arm64'],
            capture_output=True,
            text=True
        )
        return '1' in result.stdout
    except:
        return 'arm64' in platform.machine().lower()

def check_pytorch_mps():
    """Check if PyTorch is installed with MPS support"""
    try:
        import torch
        
        print(f"  PyTorch version: {torch.__version__}")
        
        if not torch.backends.mps.is_available():
            print("  WARNING: MPS not available")
            return False
            
        if not torch.backends.mps.is_built():
            print("  WARNING: PyTorch not built with MPS support")
            return False
            
        # Test MPS
        try:
            device = torch.device('mps')
            x = torch.randn(5, 5, device=device)
            y = x @ x.T
            print("  MPS test: OK")
            return True
        except Exception as e:
            print(f"  MPS test failed: {e}")
            return False
            
    except ImportError:
        print("  PyTorch not installed")
        return False

def check_dependencies():
    """Check if all required packages are installed"""
    required_packages = [
        ('google.cloud.bigquery', 'google-cloud-bigquery'),
        ('torch', 'torch'),
        ('duckdb', 'duckdb'),
        ('numpy', 'numpy'),
        ('scipy', 'scipy'),
        ('sklearn', 'scikit-learn'),
        ('tqdm', 'tqdm')
    ]
    
    missing = []
    installed = []
    
    for import_name, package_name in required_packages:
        try:
            module = __import__(import_name.split('.')[0])
            # Get version if available
            version = getattr(module, '__version__', 'unknown')
            installed.append(f"{package_name} ({version})")
        except ImportError:
            missing.append(package_name)
            
    return missing, installed

def check_gcp_credentials():
    """Check if GCP credentials file exists"""
    cred_file = Path('gcp_prod_key.json')
    
    if not cred_file.exists():
        print("ERROR: gcp_prod_key.json not found")
        print("\nPlease create gcp_prod_key.json with your service account credentials:")
        print("""
{
  "type": "service_account",
  "project_id": "your-project-id",
  "private_key_id": "...",
  "private_key": "-----BEGIN RSA PRIVATE KEY-----\\n...",
  "client_email": "service-account@project.iam.gserviceaccount.com",
  "client_id": "...",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "...",
  "client_x509_cert_url": "..."
}
        """)
        return False
        
    # Validate JSON structure
    try:
        with open(cred_file, 'r') as f:
            creds = json.load(f)
            
        required_fields = ['type', 'project_id', 'private_key', 'client_email']
        for field in required_fields:
            if field not in creds:
                print(f"ERROR: Missing field '{field}' in gcp_prod_key.json")
                return False
                
    except json.JSONDecodeError:
        print("ERROR: gcp_prod_key.json is not valid JSON")
        return False
        
    return True

def check_config():
    """Check if config.json exists and is valid"""
    config_file = Path('config.json')
    
    if not config_file.exists():
        print("Creating default config.json...")
        default_config = {
            'projects': ['your-project-id-1', 'your-project-id-2'],
            'max_workers': 5,
            'max_memory_mb': 8192,
            'rows_per_batch': 10000,
            'checkpoint_enabled': True,
            'retry_attempts': 3,
            'retry_delay_seconds': 5
        }
        
        with open(config_file, 'w') as f:
            json.dump(default_config, f, indent=2)
            
        print(f"\nPlease edit config.json with your BigQuery project IDs")
        return False
        
    # Check if projects are configured
    with open(config_file, 'r') as f:
        config = json.load(f)
        
    if config.get('projects') == ['your-project-id-1', 'your-project-id-2']:
        print("\nERROR: Please edit config.json with your actual BigQuery project IDs")
        return False
        
    return True

def print_installation_instructions():
    """Print instructions for installing dependencies"""
    print("\n" + "="*60)
    print("Installation Instructions for Mac M1/M2/M3:")
    print("="*60)
    print("\nOption 1: Run the install script")
    print("  chmod +x install.sh")
    print("  ./install.sh")
    print("\nOption 2: Manual installation")
    print("  # Install PyTorch for Apple Silicon:")
    print("  pip3 install torch torchvision torchaudio")
    print("\n  # Install other dependencies:")
    print("  pip3 install google-cloud-bigquery duckdb numpy scipy scikit-learn tqdm")
    print("\n" + "="*60)

def main():
    print("CMDB+ Environment Check")
    print("="*50)
    
    # Check Mac Silicon
    print("Checking Mac Silicon...", end=" ")
    if not check_mac_silicon():
        print("FAILED")
        print("\nERROR: Apple Silicon Mac (M1/M2/M3) required")
        print("This system requires Mac M1/M2/M3 GPU acceleration")
        sys.exit(1)
    print("OK")
    
    # Check Python version
    print("Checking Python version...", end=" ")
    if sys.version_info < (3, 8):
        print("FAILED")
        print(f"\nERROR: Python 3.8+ required (you have {sys.version})")
        sys.exit(1)
    print(f"OK ({sys.version.split()[0]})")
    
    # Check dependencies
    print("\nChecking dependencies:")
    missing, installed = check_dependencies()
    
    if installed:
        print("  Installed packages:")
        for package in installed:
            print(f"    ✓ {package}")
    
    if missing:
        print("  Missing packages:")
        for package in missing:
            print(f"    ✗ {package}")
        print_installation_instructions()
        sys.exit(1)
    
    # Check PyTorch MPS
    print("\nChecking PyTorch MPS support:")
    if not check_pytorch_mps():
        print("\nWARNING: PyTorch MPS support not available")
        print("The system requires GPU acceleration")