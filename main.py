#!/usr/bin/env python3

import asyncio
import json
import sys
import os
from pathlib import Path
from datetime import datetime
import logging
import argparse
import signal
from typing import Dict, Optional

from core.orchestrator import AdvancedOrchestrator
from core.bigquery_scanner import BigQueryScanner
from core.checkpoint_manager import CheckpointManager

from models.sherlock_model import SherlockModel
from models.sato_model import SatoModel
from models.doduo_model import DoduoModel
from models.llm_classifier import LLMClassifier
from models.ensemble_predictor import EnsemblePredictor

from processors.feature_extractor import AdvancedFeatureExtractor
from processors.entity_resolver import EntityResolver
from processors.graph_builder import GraphBuilder
from processors.context_analyzer import ContextAnalyzer

from storage.cmdb_builder import CMDBBuilder
from storage.metadata_store import MetadataStore
from storage.streaming_handler import StreamingHandler

from utils.gpu_optimizer import GPUOptimizer
from utils.data_loader import FastTabularDataLoader, CachedDataLoader

log_dir = Path('logs')
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f'cmdb_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class CMDBPlusAdvanced:
    def __init__(self, config_path: str = 'config.json', resume: bool = True):
        self.config_path = Path(config_path)
        self.resume = resume
        self.config = None
        self.orchestrator = None
        self.shutdown_requested = False
        
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        logger.info("Shutdown signal received. Saving state...")
        self.shutdown_requested = True
        if self.orchestrator:
            self.orchestrator.shutdown_requested = True
        sys.exit(0)
    
    def load_config(self) -> Dict:
        if not self.config_path.exists():
            self._create_default_config()
            logger.error(f"Created default config at {self.config_path}")
            logger.error("Please edit config.json with your BigQuery project IDs and run again")
            sys.exit(1)
        
        with open(self.config_path) as f:
            config = json.load(f)
        
        if not config.get('projects') or 'your-project-id' in str(config['projects']):
            logger.error("Please update config.json with actual BigQuery project IDs")
            sys.exit(1)
        
        self._validate_config(config)
        return config
    
    def _create_default_config(self):
        default_config = {
            "projects": ["your-project-id-1", "your-project-id-2"],
            "max_workers": 8,
            "batch_size": 128,
            "device": "mps",
            "model_config": {
                "sherlock_features": 1588,
                "sato_topics": 400,
                "doduo_tokens": 8,
                "ensemble_threshold": 0.85,
                "llm_model": "gpt-4",
                "llm_batch_size": 50,
                "weak_supervision_ratio": 0.013
            },
            "entity_resolution": {
                "blocking_strategy": "neural_lsh",
                "comparison_threshold": 0.01,
                "graph_clustering": True,
                "confidence_levels": [0.9, 0.7, 0.5]
            },
            "storage": {
                "streaming_enabled": False,
                "kafka_brokers": ["localhost:9092"],
                "elasticsearch_host": "localhost:9200",
                "neo4j_uri": "bolt://localhost:7687",
                "duckdb_path": "cmdb.db"
            },
            "optimization": {
                "mixed_precision": True,
                "gradient_checkpointing": True,
                "memory_fraction": 0.9,
                "cache_size_gb": 8
            },
            "sampling": {
                "max_rows_per_table": 100000,
                "column_sample_size": 1000,
                "confidence_threshold": 0.7
            }
        }
        
        with open(self.config_path, 'w') as f:
            json.dump(default_config, f, indent=2)
    
    def _validate_config(self, config: Dict):
        required_keys = ['projects', 'max_workers', 'batch_size']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required config key: {key}")
        
        if not isinstance(config['projects'], list) or len(config['projects']) == 0:
            raise ValueError("'projects' must be a non-empty list")
        
        if config['max_workers'] < 1:
            raise ValueError("'max_workers' must be at least 1")
    
    def check_gcp_credentials(self):
        cred_paths = [
            Path('gcp_prod_key.json'),
            Path(os.environ.get('GOOGLE_APPLICATION_CREDENTIALS', ''))
        ]
        
        for path in cred_paths:
            if path and path.exists():
                logger.info(f"Found GCP credentials at {path}")
                return True
        
        if 'GOOGLE_APPLICATION_CREDENTIALS' in os.environ:
            logger.info("Using GOOGLE_APPLICATION_CREDENTIALS environment variable")
            return True
        
        logger.warning("No GCP credentials found. Trying default credentials...")
        return True
    
    def initialize_gpu(self) -> str:
        logger.info("Initializing GPU...")
        
        gpu_optimizer = GPUOptimizer()
        
        try:
            device = gpu_optimizer.initialize()
            
            if device != 'mps':
                raise RuntimeError("MPS device not available")
            
            logger.info(f"GPU initialized successfully: {device}")
            logger.info(f"Memory info: {gpu_optimizer.get_memory_info()}")
            
            return device
            
        except Exception as e:
            logger.error(f"GPU initialization failed: {e}")
            logger.error("This system requires Mac M1/M2/M3 with MPS support")
            sys.exit(1)
    
    async def run(self):
        logger.info("="*60)
        logger.info("CMDB+ Advanced - AI-Powered Configuration Management")
        logger.info("="*60)
        
        self.config = self.load_config()
        logger.info(f"Loaded configuration for {len(self.config['projects'])} projects")
        
        if not self.check_gcp_credentials():
            logger.error("No valid GCP credentials found")
            sys.exit(1)
        
        device = self.initialize_gpu()
        
        logger.info("Initializing orchestrator...")
        self.orchestrator = AdvancedOrchestrator(self.config, device)
        
        if not self.resume:
            logger.info("Resume disabled, clearing checkpoints...")
            checkpoint_manager = CheckpointManager()
            checkpoint_manager.clear()
        
        logger.info("Starting discovery process...")
        start_time = datetime.now()
        
        try:
            await self.orchestrator.execute()
            
            elapsed = (datetime.now() - start_time).total_seconds()
            logger.info(f"Discovery completed in {elapsed:.1f} seconds")
            
            self._print_summary()
            
        except KeyboardInterrupt:
            logger.info("Process interrupted by user")
            self._save_state()
            
        except Exception as e:
            logger.error(f"Fatal error: {e}", exc_info=True)
            self._save_state()
            sys.exit(1)
    
    def _save_state(self):
        if self.orchestrator:
            checkpoint_manager = CheckpointManager()
            checkpoint_manager.save({
                'hosts': self.orchestrator.discovered_hosts,
                'metadata': self.orchestrator.column_metadata,
                'statistics': self.orchestrator.statistics
            })
            logger.info("State saved to checkpoint")
    
    def _print_summary(self):
        if not self.orchestrator:
            return
        
        stats = self.orchestrator.statistics
        
        print("\n" + "="*60)
        print("DISCOVERY SUMMARY")
        print("="*60)
        print(f"Tables Processed:     {stats['tables_processed']:,}")
        print(f"Rows Scanned:         {stats['rows_scanned']:,}")
        print(f"Hosts Discovered:     {stats['hosts_discovered']:,}")
        print(f"Columns Classified:   {stats['columns_classified']:,}")
        print(f"Entities Resolved:    {stats['entities_resolved']:,}")
        print("="*60)
        
        cmdb = CMDBBuilder(self.config['storage']['duckdb_path'])
        db_stats = cmdb.get_statistics()
        
        if db_stats:
            print("\nCMDB DATABASE STATISTICS:")
            print(f"Total Hosts:          {db_stats.get('total_hosts', 0):,}")
            print(f"High Confidence:      {db_stats.get('high_confidence', 0):,}")
            print(f"Relationships:        {db_stats.get('total_relationships', 0):,}")
            print(f"Avg Quality Score:    {db_stats.get('avg_quality_score', 0):.2f}")
        
        print("\nDatabase Location:    ", self.config['storage']['duckdb_path'])
        print("Query with:           duckdb", self.config['storage']['duckdb_path'])
        print("="*60)

async def main():
    parser = argparse.ArgumentParser(description='CMDB+ Advanced Discovery System')
    parser.add_argument(
        '--config',
        type=str,
        default='config.json',
        help='Path to configuration file (default: config.json)'
    )
    parser.add_argument(
        '--no-resume',
        action='store_true',
        help='Start fresh without resuming from checkpoint'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    app = CMDBPlusAdvanced(
        config_path=args.config,
        resume=not args.no_resume
    )
    
    await app.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unhandled exception: {e}", exc_info=True)
        sys.exit(1)