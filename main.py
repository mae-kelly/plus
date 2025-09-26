#!/usr/bin/env python3

import asyncio
import json
from pathlib import Path
from core.orchestrator import AdvancedOrchestrator
from utils.gpu_optimizer import GPUOptimizer
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def main():
    config_path = Path('config.json')
    
    if not config_path.exists():
        raise FileNotFoundError("config.json not found")
    
    with open(config_path) as f:
        config = json.load(f)
    
    if not config['projects'] or 'your-project-id' in str(config['projects']):
        raise ValueError("Update config.json with actual project IDs")
    
    gpu_optimizer = GPUOptimizer()
    device = gpu_optimizer.initialize()
    
    if device != 'mps':
        raise RuntimeError("Mac M1/M2/M3 with MPS required")
    
    orchestrator = AdvancedOrchestrator(config, device)
    await orchestrator.execute()

if __name__ == "__main__":
    asyncio.run(main())