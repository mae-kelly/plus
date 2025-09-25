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

def check_dependencies():
    """Check if all required packages are installed"""
    required_packages = [
        'google-cloud-bigquery',
        'torch',
        'duckdb',
        'numpy',
        'scipy',
        'sklearn',
        'tqdm'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing.append(package)
            
    return missing

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
    print("Checking dependencies...", end=" ")
    missing = check_dependencies()
    if missing:
        print("FAILED")
        print(f"\nMissing packages: {', '.join(missing)}")
        print("\nInstall with:")
        print(f"  pip install -r requirements.txt")
        sys.exit(1)
    print("OK")
    
    # Check GCP credentials
    print("Checking GCP credentials...", end=" ")
    if not check_gcp_credentials():
        sys.exit(1)
    print("OK")
    
    # Check config
    print("Checking configuration...", end=" ")
    if not check_config():
        sys.exit(1)
    print("OK")
    
    print("\n" + "="*50)
    print("All checks passed! You can now run:")
    print("  python main.py")
    
    # Create necessary directories
    Path('logs').mkdir(exist_ok=True)
    print("\nCreated logs directory")

if __name__ == "__main__":
    main()