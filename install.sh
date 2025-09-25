#!/bin/bash

echo "CMDB+ PyTorch Setup for Mac M1/M2/M3"
echo "====================================="

# Detect Python version
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Detected Python version: $PYTHON_VERSION"

# Method 1: If you want to use Homebrew Python + pip
echo ""
echo "Method 1: Installing PyTorch via pip (Recommended)"
echo "---------------------------------------------------"

# Uninstall any conflicting versions first
pip3 uninstall torch torchvision torchaudio -y 2>/dev/null

# Install PyTorch for Mac M1/M2/M3
pip3 install --upgrade pip
pip3 install torch torchvision torchaudio

# Test installation
python3 -c "
import torch
print(f'✓ PyTorch {torch.__version__} installed')
print(f'✓ MPS available: {torch.backends.mps.is_available()}')
" || echo "❌ PyTorch installation failed"

echo ""
echo "Method 2: If you prefer using Homebrew's PyTorch"
echo "-------------------------------------------------"
echo "You need to ensure Python can find Homebrew's PyTorch:"

# Find where Homebrew installed PyTorch
BREW_PREFIX=$(brew --prefix)
PYTORCH_PATH="${BREW_PREFIX}/lib/python${PYTHON_VERSION}/site-packages"

echo "Add this to your shell profile (~/.zshrc or ~/.bash_profile):"
echo "export PYTHONPATH=\"${PYTORCH_PATH}:\$PYTHONPATH\""

# Or create a .pth file
echo ""
echo "Or create a .pth file to link Homebrew's PyTorch:"
SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")
echo "sudo echo '${PYTORCH_PATH}' > ${SITE_PACKAGES}/homebrew-pytorch.pth"