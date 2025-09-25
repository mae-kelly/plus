import os
import platform
import torch
import logging

logger = logging.getLogger(__name__)

class GPUAccelerator:
    
    def __init__(self):
        self.device = None
        self.backend = None
        
    def initialize(self) -> str:
        # Check we're on Mac
        if platform.system() != 'Darwin':
            raise RuntimeError("Mac required for M1 GPU. No CPU fallback allowed.")
            
        # Check for Apple Silicon
        if not self._is_apple_silicon():
            raise RuntimeError("Apple Silicon (M1/M2/M3) required. No CPU fallback allowed.")
            
        # Try MPS backend
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS backend not available. No CPU fallback allowed.")
            
        if not torch.backends.mps.is_built():
            raise RuntimeError("PyTorch not built with MPS support. No CPU fallback allowed.")
            
        # Test MPS
        try:
            device = torch.device("mps")
            test_tensor = torch.randn(10, 10, device=device)
            _ = test_tensor @ test_tensor.T
            
            # Optimize settings
            self._optimize_mps()
            
            logger.info("GPU initialized: Mac M1 MPS")
            self.device = "mps"
            self.backend = "mps"
            
            return "mps"
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize M1 GPU: {e}. No CPU fallback allowed.")
            
    def _is_apple_silicon(self) -> bool:
        try:
            import subprocess
            result = subprocess.run(
                ['sysctl', '-n', 'hw.optional.arm64'],
                capture_output=True,
                text=True
            )
            return '1' in result.stdout
        except:
            # Check architecture
            return 'arm64' in platform.machine().lower()
            
    def _optimize_mps(self):
        # MPS memory optimization
        os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
        os.environ['PYTORCH_MPS_LOW_WATERMARK_RATIO'] = '0.0'
        os.environ['PYTORCH_MPS_ALLOCATOR_POLICY'] = 'garbage_collection'
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'  # No fallback to CPU
        
        # Enable fast math
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('high')
        
        logger.info("MPS optimizations applied")
        
    def get_device_info(self) -> dict:
        return {
            'device': self.device,
            'backend': self.backend,
            'mps_available': torch.backends.mps.is_available(),
            'mps_built': torch.backends.mps.is_built()
        }