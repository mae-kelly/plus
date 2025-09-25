#!/usr/bin/env python3
"""
CMDB+ Setup Script with Homebrew PyTorch support
"""

import sys
import os
import platform
import subprocess
import json
from pathlib import Path

def find_torch():
    """Try to find and import torch from various locations"""
    import_successful = False
    
    # Method 1: Try standard import
    try:
        import torch
        print(f"  ✓ Found torch via standard import")
        print(f"    Version: {torch.__version__}")
        print(f"    Location: {torch.__file__}")
        return True
    except ImportError:
        pass
    
    # Method 2: Check Homebrew paths
    brew_prefix = subprocess.run(['brew', '--prefix'], 
                                capture_output=True, 
                                text=True).stdout.strip()
    
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    possible_paths = [
        f"{brew_prefix}/lib/python{python_version}/site-packages",
        f"/opt/homebrew/lib/python{python_version}/site-packages",
        f"/usr/local/lib/python{python_version}/site-packages",
    ]
    
    for path in possible_paths:
        if os.path.exists(path) and path not in sys.path:
            sys.path.insert(0, path)
            try:
                import torch
                print(f"  ✓ Found torch via Homebrew path: {path}")
                print(f"    Version: {torch.__version__}")
                return True
            except ImportError:
                continue
    
    # Method 3: Try to find via brew info
    try:
        result = subprocess.run(['brew', 'list', 'pytorch'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"  ℹ PyTorch is installed via Homebrew but Python can't import it")
            print(f"    Run: ./setup_pytorch.sh to fix this")
            return False
    except:
        pass
    
    print("  ✗ PyTorch not found")
    return False

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
        ('google.cloud.bigquery', 'google-cloud-bigquery'),
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
            version = getattr(module, '__version__', 'unknown')
            installed.append(f"{package_name} ({version})")
        except ImportError:
            missing.append(package_name)
            
    return missing, installed

def main():
    print("CMDB+ Environment Check")
    print("="*50)
    
    # Check Mac Silicon
    print("Checking Mac Silicon...", end=" ")
    if not check_mac_silicon():
        print("FAILED")
        print("\nERROR: Apple Silicon Mac (M1/M2/M3) required")
        sys.exit(1)
    print("OK")
    
    # Check Python version
    print("Checking Python version...", end=" ")
    if sys.version_info < (3, 8):
        print("FAILED")
        print(f"\nERROR: Python 3.8+ required (you have {sys.version})")
        sys.exit(1)
    print(f"OK ({sys.version.split()[0]})")
    
    # Check PyTorch
    print("\nChecking PyTorch:")
    if not find_torch():
        print("\n" + "="*50)
        print("PyTorch Installation Required")
        print("="*50)
        print("\nOption 1: Install via pip (Recommended)")
        print("  pip3 install torch torchvision torchaudio")
        print("\nOption 2: Fix Homebrew PyTorch")
        print("  chmod +x setup_pytorch.sh")
        print("  ./setup_pytorch.sh")
        sys.exit(1)
    
    # Now test MPS support
    try:
        import torch
        if torch.backends.mps.is_available():
            print(f"  ✓ MPS support available")
            # Test it works
            device = torch.device('mps')
            x = torch.randn(5, 5, device=device)
            print(f"  ✓ MPS test successful")
        else:
            print(f"  ✗ MPS not available")
            sys.exit(1)
    except Exception as e:
        print(f"  ✗ MPS test failed: {e}")
        sys.exit(1)
    
    # Check other dependencies
    print("\nChecking other dependencies:")
    missing, installed = check_dependencies()
    
    if missing:
        print(f"  Missing: {', '.join(missing)}")
        print(f"\n  Install with:")
        print(f"    pip3 install {' '.join(missing)}")
        sys.exit(1)
    else:
        print(f"  ✓ All dependencies installed")
    
    # Check GCP credentials
    print("\nChecking GCP credentials...", end=" ")
    cred_file = Path('gcp_prod_key.json')
    if not cred_file.exists():
        print("FAILED")
        print("\n  Create gcp_prod_key.json with your service account key")
        sys.exit(1)
    print("OK")
    
    # Check config
    print("Checking configuration...", end=" ")
    config_file = Path('config.json')
    if not config_file.exists():
        with open(config_file, 'w') as f:
            json.dump({
                'projects': ['your-project-id-1', 'your-project-id-2'],
                'max_workers': 5,
                'max_memory_mb': 8192,
                'rows_per_batch': 10000,
                'checkpoint_enabled': True,
                'retry_attempts': 3,
                'retry_delay_seconds': 5
            }, f, indent=2)
        print("CREATED")
        print("\n  Edit config.json with your BigQuery project IDs")
        sys.exit(1)
    print("OK")
    
    print("\n" + "="*50)
    print("✅ All checks passed!")
    print("\nRun: python3 main.py")
    
    # Create directories
    Path('logs').mkdir(exist_ok=True)

if __name__ == "__main__":
    main()