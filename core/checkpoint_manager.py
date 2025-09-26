import pickle
import json
from pathlib import Path
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class CheckpointManager:
    def __init__(self, checkpoint_dir: str = 'checkpoints'):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.checkpoint_file = self.checkpoint_dir / 'cmdb_checkpoint.pkl'
        self.metadata_file = self.checkpoint_dir / 'metadata.json'
        
    def save(self, state: dict):
        try:
            temp_file = self.checkpoint_file.with_suffix('.tmp')
            
            with open(temp_file, 'wb') as f:
                pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            temp_file.replace(self.checkpoint_file)
            
            metadata = {
                'timestamp': datetime.now().isoformat(),
                'statistics': state.get('statistics', {}),
                'hosts_count': len(state.get('hosts', {})),
                'metadata_count': len(state.get('metadata', {}))
            }
            
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
                
            logger.debug(f"Checkpoint saved: {metadata['hosts_count']} hosts")
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
    
    def load(self):
        if not self.checkpoint_file.exists():
            return None
        
        try:
            with open(self.checkpoint_file, 'rb') as f:
                state = pickle.load(f)
            
            logger.info("Checkpoint loaded successfully")
            return state
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None
    
    def clear(self):
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()
        if self.metadata_file.exists():
            self.metadata_file.unlink()
        logger.info("Checkpoints cleared")