#!/bin/bash

echo "CMDB+ Installation Script for Mac M1/M2/M3"
echo "=========================================="

# Check if we're on Mac
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "Error: This script is for macOS only"
    exit 1
fi

# Check for Apple Silicon
if [[ $(uname -m) != "arm64" ]]; then
    echo "Warning: Not running on Apple Silicon (M1/M2/M3)"
    echo "Continue anyway? (y/n)"
    read -r response
    if [[ "$response" != "y" ]]; then
        exit 1
    fi
fi

echo ""
echo "Step 1: Installing PyTorch for Apple Silicon..."
echo "------------------------------------------------"

# Install PyTorch for Mac M1/M2/M3
pip3 install --upgrade pip
pip3 install torch torchvision torchaudio

echo ""
echo "Step 2: Installing other dependencies..."
echo "------------------------------------------------"

# Install other requirements
pip3 install google-cloud-bigquery
pip3 install duckdb
pip3 install numpy
pip3 install scipy
pip3 install scikit-learn
pip3 install tqdm

echo ""
echo "Step 3: Verifying installations..."
echo "------------------------------------------------"

# Verify PyTorch MPS support
python3 -c "
import torch
import platform

print(f'Python version: {platform.python_version()}')
print(f'PyTorch version: {torch.__version__}')
print(f'MPS Available: {torch.backends.mps.is_available()}')
print(f'MPS Built: {torch.backends.mps.is_built()}')

if torch.backends.mps.is_available():
    # Test MPS
    device = torch.device('mps')
    x = torch.randn(5, 5, device=device)
    y = x @ x.T
    print('MPS test successful!')
else:
    print('WARNING: MPS not available!')
"

echo ""
echo "Installation complete!"
echo ""
echo "Next steps:"
echo "1. Create gcp_prod_key.json with your service account credentials"
echo "2. Edit config.json with your BigQuery project IDs"
echo "3. Run: python3 main.py"