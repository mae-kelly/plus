import os
import platform
import torch
import logging

logger = logging.getLogger(__name__)

class GPUOptimizer:
    def __init__(self):
        self.device = None
        self.optimization_applied = False
    
    def initialize(self) -> str:
        if platform.system() != 'Darwin':
            raise RuntimeError("Mac required for MPS acceleration")
        
        if not self._is_apple_silicon():
            raise RuntimeError("Apple Silicon (M1/M2/M3) required")
        
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS backend not available")
        
        if not torch.backends.mps.is_built():
            raise RuntimeError("PyTorch not built with MPS support")
        
        try:
            device = torch.device("mps")
            
            test_tensor = torch.randn(100, 100, device=device)
            result = test_tensor @ test_tensor.T
            del test_tensor, result
            
            self._apply_optimizations()
            
            self.device = "mps"
            logger.info("GPU initialized: Mac MPS backend")
            
            return "mps"
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize MPS: {e}")
    
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
            return 'arm64' in platform.machine().lower()
    
    def _apply_optimizations(self):
        os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
        os.environ['PYTORCH_MPS_LOW_WATERMARK_RATIO'] = '0.0'
        os.environ['PYTORCH_MPS_ALLOCATOR_POLICY'] = 'garbage_collection'
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
        
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('high')
        
        self.optimization_applied = True
        logger.info("MPS optimizations applied")
    
    def get_memory_info(self) -> dict:
        if self.device != "mps":
            return {}
        
        return {
            'device': self.device,
            'backend': 'Metal Performance Shaders',
            'unified_memory': True,
            'optimizations': self.optimization_applied
        }
    
    def optimize_model(self, model: torch.nn.Module) -> torch.nn.Module:
        model = model.to(self.device)
        
        if hasattr(torch, 'compile') and self.device == 'mps':
            try:
                model = torch.compile(model, mode='reduce-overhead')
                logger.info("Model compiled with torch.compile")
            except:
                logger.debug("torch.compile not available")
        
        return model
    
    def enable_mixed_precision(self):
        return torch.cuda.amp.autocast(device_type='mps', dtype=torch.float16)
    
    def create_scaler(self):
        return torch.cuda.amp.GradScaler('mps')