#!/usr/bin/env python3
"""
CMDB+ Main Orchestrator - No Fallbacks Version
Mac M1/M2/M3 GPU Required - No CPU fallback
Handles all BigQuery issues encountered
"""

import os
import sys
import json
import asyncio
import pickle
import signal
from pathlib import Path
from typing import Dict, List, Set, Any, Optional, Tuple
from collections import defaultdict
from datetime import datetime, timedelta
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
from google.cloud import bigquery
from google.api_core import exceptions
import time
import hashlib
import re
import traceback
import psutil

# Setup logging
log_dir = Path('logs')
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f'cmdb_plus_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Suppress noisy Google Cloud logging
logging.getLogger('google.cloud').setLevel(logging.WARNING)
logging.getLogger('google.auth').setLevel(logging.WARNING)

class CMDBPlusOrchestrator:
    
    def __init__(self, project_ids: List[str], config: Dict[str, Any] = None):
        self.project_ids = project_ids
        self.config = config or self._load_config()
        
        # Cache directory
        self.cache_dir = Path('cache')
        self.cache_dir.mkdir(exist_ok=True)
        
        # Files
        today = datetime.now().strftime("%Y%m%d")
        self.daily_cache_file = self.cache_dir / f'cmdb_cache_{today}.pkl'
        self.checkpoint_file = Path('cmdb_checkpoint.pkl')
        
        # Tracking
        self.discovered_hosts = {}
        self.column_importance = defaultdict(int)
        self.processed_tables = set()
        self.failed_tables = set()
        self.dataset_locations = {}
        self.problematic_columns = defaultdict(set)
        
        # Metrics
        self.scan_metrics = {
            'projects': 0, 'datasets': 0, 'tables': 0, 
            'tables_skipped': 0, 'tables_failed': 0,
            'rows': 0, 'hosts': 0, 'errors': 0,
            'start_time': datetime.now()
        }
        
        # Patterns to skip
        self.skip_table_patterns = [
            r'.*_staging$', r'.*_temp$', r'.*_tmp$', r'.*_backup$',
            r'.*_test$', r'.*_old$', r'^temp_', r'^tmp_',
            r'.*_\d{8}$', r'.*_\d{14}$'  # Date suffixed tables
        ]
        
        # Complex column types to skip
        self.skip_column_types = ['RECORD', 'STRUCT', 'ARRAY', 'GEOGRAPHY', 'JSON', 'BYTES', 'REPEATED']
        
        # Query settings
        self.batch_size = self.config.get('rows_per_batch', 5000)
        self.max_memory_percent = 80
        
        # Shutdown handling
        self.shutdown_requested = False
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Initialize components
        self._initialize_gpu()
        self._initialize_components()
        
        # Thread pool
        self.executor = ThreadPoolExecutor(max_workers=self.config.get('max_workers', 3))
        
        # Load any existing cache/checkpoint
        self.using_cache = self._load_cache_or_checkpoint()
        
    def _load_config(self) -> Dict[str, Any]:
        config_file = Path('config.json')
        
        default_config = {
            'projects': ['your-project-id'],
            'max_workers': 3,
            'rows_per_batch': 5000,
            'checkpoint_enabled': True,
            'use_daily_cache': True,
            'cache_expiry_hours': 24,
            'max_training_samples': 10000,
            'training_timeout_seconds': 120,
            'column_limit': 100,
            'max_raw_forms_per_host': 20,
            'max_occurrences_per_host': 100,
            'max_values_per_column': 50
        }
        
        if config_file.exists():
            with open(config_file, 'r') as f:
                loaded = json.load(f)
                default_config.update(loaded)
                return default_config
        else:
            with open(config_file, 'w') as f:
                json.dump(default_config, f, indent=2)
            logger.info("Created config.json - edit with your project IDs")
            return default_config
            
    def _signal_handler(self, signum, frame):
        logger.info("Shutdown signal received. Saving state...")
        self.shutdown_requested = True
        self._save_checkpoint()
        if self.executor:
            self.executor.shutdown(wait=False)
        sys.exit(0)
        
    def _initialize_gpu(self):
        """Initialize GPU - no fallback to CPU"""
        from gpu_accelerator import GPUAccelerator
        self.gpu = GPUAccelerator()
        self.device = self.gpu.initialize()
        
        if 'cpu' in str(self.device).lower():
            logger.error("GPU required. Mac M1/M2/M3 with MPS support needed.")
            sys.exit(1)
            
        logger.info(f"GPU initialized: {self.device}")
        
    def _initialize_components(self):
        logger.info("Initializing components...")
        
        from bigquery_client import BigQueryClientManager
        from discovery_engine import DiscoveryEngine
        from normalization_engine import NormalizationEngine
        from pattern_recognition import PatternRecognitionModel
        from column_classifier import ColumnClassifier
        from data_aggregator import DataAggregator
        from cmdb_builder import CMDBBuilder
        from relationship_analyzer import RelationshipAnalyzer
        
        self.bq_manager = BigQueryClientManager()
        self.discovery_engine = DiscoveryEngine(device=self.device)
        self.normalizer = NormalizationEngine()
        self.pattern_model = PatternRecognitionModel(device=self.device)
        self.column_classifier = ColumnClassifier(device=self.device)
        self.aggregator = DataAggregator()
        self.relationship_analyzer = RelationshipAnalyzer()
        self.cmdb_builder = CMDBBuilder('new_cmdb.db')
        
        logger.info("Components initialized")
        
    def _load_cache_or_checkpoint(self) -> bool:
        # Try daily cache
        if self.daily_cache_file.exists() and self.config.get('use_daily_cache', True):
            try:
                with open(self.daily_cache_file, 'rb') as f:
                    cache = pickle.load(f)
                    
                cache_time = cache.get('timestamp', datetime.min)
                age_hours = (datetime.now() - cache_time).total_seconds() / 3600
                
                if age_hours < self.config.get('cache_expiry_hours', 24):
                    self.discovered_hosts = cache.get('discovered_hosts', {})
                    self.column_importance = cache.get('column_importance', defaultdict(int))
                    self.processed_tables = cache.get('processed_tables', set())
                    self.failed_tables = cache.get('failed_tables', set())
                    
                    logger.info(f"Cache from {cache_time.strftime('%Y-%m-%d %H:%M')}: {len(self.discovered_hosts)} hosts")
                    
                    response = input(f"Use cached data? (y/n): ").strip().lower()
                    if response != 'n':
                        return True
                        
            except Exception as e:
                logger.debug(f"Cache load failed: {e}")
                
        # Try checkpoint
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'rb') as f:
                    checkpoint = pickle.load(f)
                    
                self.discovered_hosts = checkpoint.get('discovered_hosts', {})
                self.column_importance = checkpoint.get('column_importance', defaultdict(int))
                self.processed_tables = checkpoint.get('processed_tables', set())
                self.failed_tables = checkpoint.get('failed_tables', set())
                
                logger.info(f"Resumed checkpoint: {len(self.processed_tables)} tables done")
                
            except Exception as e:
                logger.debug(f"Checkpoint load failed: {e}")
                
        return False
        
    def _save_checkpoint(self):
        checkpoint = {
            'discovered_hosts': self.discovered_hosts,
            'column_importance': dict(self.column_importance),
            'processed_tables': self.processed_tables,
            'failed_tables': self.failed_tables,
            'timestamp': datetime.now()
        }
        
        try:
            temp = self.checkpoint_file.with_suffix('.tmp')
            with open(temp, 'wb') as f:
                pickle.dump(checkpoint, f)
            temp.replace(self.checkpoint_file)
            logger.debug("Checkpoint saved")
        except Exception as e:
            logger.error(f"Checkpoint save failed: {e}")
            
    def _save_daily_cache(self):
        cache = {
            'discovered_hosts': self.discovered_hosts,
            'column_importance': dict(self.column_importance),
            'processed_tables': self.processed_tables,
            'failed_tables': self.failed_tables,
            'timestamp': datetime.now()
        }
        
        try:
            with open(self.daily_cache_file, 'wb') as f:
                pickle.dump(cache, f)
            logger.info("Daily cache saved")
            
            # Clean old caches
            for f in self.cache_dir.glob('cmdb_cache_*.pkl'):
                if (datetime.now() - datetime.fromtimestamp(f.stat().st_mtime)).days > 7:
                    f.unlink()
                    
        except Exception as e:
            logger.error(f"Cache save failed: {e}")
            
    def _check_memory(self) -> bool:
        memory = psutil.virtual_memory()
        if memory.percent > self.max_memory_percent:
            logger.warning(f"Memory at {memory.percent}% - cleaning up")
            gc.collect()
            return psutil.virtual_memory().percent < self.max_memory_percent
        return True
        
    def _should_skip_table(self, table_id: str) -> bool:
        for pattern in self.skip_table_patterns:
            if re.match(pattern, table_id, re.IGNORECASE):
                return True
        return False
        
    def _get_safe_columns(self, schema) -> List[str]:
        safe = []
        for field in schema:
            if field.field_type not in self.skip_column_types:
                safe.append(field.name)
            else:
                logger.debug(f"Skipping {field.name} (type: {field.field_type})")
        return safe
        
    def _execute_query(self, client, query: str, timeout: int = 30) -> Optional[List]:
        job_config = bigquery.QueryJobConfig()
        job_config.use_query_cache = True
        job_config.use_legacy_sql = False
        
        try:
            job = client.query(query, job_config=job_config)
            return list(job.result(timeout=timeout))
            
        except exceptions.BadRequest as e:
            error = str(e)
            
            # Handle partitioned tables
            if 'without a filter' in error.lower() and 'WHERE' not in query:
                modified = query.replace('LIMIT', 'WHERE DATE(_PARTITIONTIME) >= CURRENT_DATE() - 7 LIMIT')
                return self._execute_query(client, modified, timeout)
                
            logger.debug(f"Bad query: {error[:100]}")
            return None
            
        except exceptions.Forbidden:
            logger.debug("Permission denied")
            return None
            
        except exceptions.NotFound:
            logger.debug("Not found")
            return None
            
        except Exception as e:
            logger.debug(f"Query error: {str(e)[:100]}")
            return None
            
    async def execute(self):
        logger.info(f"Starting CMDB+ for: {self.project_ids}")
        
        if self.using_cache:
            logger.info("Using cache - skipping to build")
            await self._build_cmdb()
            self._report()
            return
            
        try:
            # Phase 1: Quick training
            await self._train()
            
            # Phase 2: Discovery
            await self._discover()
            
            # Phase 3: Analysis
            await self._analyze()
            
            # Save cache
            self._save_daily_cache()
            
            # Phase 4: Build
            await self._build_cmdb()
            
            self._report()
            
            # Cleanup
            if self.checkpoint_file.exists():
                self.checkpoint_file.unlink()
                
        except Exception as e:
            logger.error(f"Execution failed: {e}")
            self._save_checkpoint()
            raise
        finally:
            if self.executor:
                self.executor.shutdown(wait=True)
                
    async def _train(self):
        logger.info("Phase 1: Training...")
        
        samples = []
        start = time.time()
        timeout = self.config.get('training_timeout_seconds', 120)
        
        for project_id in self.project_ids[:1]:  # First project only
            if time.time() - start > timeout:
                break
                
            try:
                client = self.bq_manager.get_client(project_id)
                datasets = list(client.list_datasets(max_results=5))
                
                for dataset in datasets[:3]:
                    tables = list(client.list_tables(dataset.reference, max_results=10))
                    
                    for table in tables[:5]:
                        if self._should_skip_table(table.table_id):
                            continue
                            
                        try:
                            t = client.get_table(table.reference)
                            
                            # Find hostname columns
                            for field in t.schema:
                                if field.field_type in self.skip_column_types:
                                    continue
                                    
                                name_lower = field.name.lower()
                                if any(x in name_lower for x in ['host', 'server', 'machine', 'node']):
                                    
                                    query = f"""
                                    SELECT DISTINCT `{field.name}`
                                    FROM `{project_id}.{dataset.dataset_id}.{table.table_id}`
                                    WHERE `{field.name}` IS NOT NULL
                                    LIMIT 100
                                    """
                                    
                                    results = self._execute_query(client, query, 10)
                                    if results:
                                        for row in results:
                                            val = getattr(row, field.name, None)
                                            if val:
                                                samples.append(str(val))
                                                
                        except Exception as e:
                            logger.debug(f"Skip {table.table_id}: {e}")
                            
            except Exception as e:
                logger.error(f"Training error: {e}")
                
        if samples:
            logger.info(f"Training with {len(samples)} samples")
            await self.discovery_engine.train(samples)
            await self.pattern_model.learn_patterns(samples)
        else:
            logger.warning("No samples - using heuristics")
            
        logger.info(f"Training done in {time.time()-start:.1f}s")
        
    async def _discover(self):
        logger.info("Phase 2: Discovery...")
        
        for project_id in self.project_ids:
            if self.shutdown_requested:
                break
                
            try:
                client = self.bq_manager.get_client(project_id)
                self.scan_metrics['projects'] += 1
                
                datasets = list(client.list_datasets())
                logger.info(f"Processing {len(datasets)} datasets in {project_id}")
                
                for dataset_ref in datasets:
                    if self.shutdown_requested:
                        break
                        
                    self.scan_metrics['datasets'] += 1
                    dataset_id = dataset_ref.dataset_id
                    
                    try:
                        tables = list(client.list_tables(dataset_ref))
                        
                        for table_ref in tables:
                            if self.shutdown_requested:
                                break
                                
                            table_id = table_ref.table_id
                            full_name = f"{project_id}.{dataset_id}.{table_id}"
                            
                            if full_name in self.processed_tables:
                                continue
                                
                            if full_name in self.failed_tables:
                                continue
                                
                            if self._should_skip_table(table_id):
                                self.scan_metrics['tables_skipped'] += 1
                                continue
                                
                            self._process_table(client, project_id, dataset_id, table_id, full_name)
                            
                    except Exception as e:
                        logger.error(f"Dataset {dataset_id} error: {e}")
                        
            except Exception as e:
                logger.error(f"Project {project_id} error: {e}")
                
    def _process_table(self, client, project_id: str, dataset_id: str, 
                      table_id: str, full_name: str):
        try:
            self.scan_metrics['tables'] += 1
            
            table = client.get_table(f"{project_id}.{dataset_id}.{table_id}")
            total_rows = table.num_rows or 0
            
            if total_rows == 0:
                self.processed_tables.add(full_name)
                return
                
            logger.info(f"Table {self.scan_metrics['tables']}: {full_name} ({total_rows:,} rows)")
            
            # Get safe columns
            safe_cols = self._get_safe_columns(table.schema)
            
            # Process in batches
            offset = 0
            failures = 0
            
            while offset < total_rows and failures < 3:
                if self.shutdown_requested:
                    break
                    
                if not self._check_memory():
                    logger.warning("Memory limit reached")
                    break
                    
                # Build query
                if safe_cols:
                    cols = ', '.join([f"`{c}`" for c in safe_cols])
                    query = f"""
                    SELECT {cols}
                    FROM `{project_id}.{dataset_id}.{table_id}`
                    LIMIT {self.batch_size}
                    OFFSET {offset}
                    """
                else:
                    query = f"""
                    SELECT *
                    FROM `{project_id}.{dataset_id}.{table_id}`
                    LIMIT {self.batch_size}
                    OFFSET {offset}
                    """
                    
                results = self._execute_query(client, query, 60)
                
                if results is None:
                    failures += 1
                    logger.warning(f"Query failed for {table_id}")
                    if failures >= 3:
                        self.failed_tables.add(full_name)
                        self.scan_metrics['tables_failed'] += 1
                        break
                    continue
                    
                failures = 0
                count = 0
                
                for row in results:
                    self.scan_metrics['rows'] += 1
                    count += 1
                    self._process_row(dict(row), full_name)
                    
                    if self.scan_metrics['rows'] % 1000 == 0:
                        self._save_checkpoint()
                        
                offset += count
                
                if count < self.batch_size:
                    break  # End of table
                    
                logger.debug(f"Processed {offset}/{total_rows} from {table_id}")
                
            self.processed_tables.add(full_name)
            
        except Exception as e:
            logger.error(f"Failed {full_name}: {e}")
            self.failed_tables.add(full_name)
            self.scan_metrics['tables_failed'] += 1
            
    def _process_row(self, row: Dict[str, Any], table: str):
        hosts = []
        normalized_row = {}
        
        for col, val in row.items():
            if val is None:
                continue
                
            self.column_importance[col] += 1
            self.column_classifier.observe(col, val, table)
            
            normalized = self.normalizer.normalize(str(val))
            raw = str(val)
            
            normalized_row[col] = {'raw': raw, 'normalized': normalized}
            
            if self.discovery_engine.is_hostname(normalized):
                conf = self.discovery_engine.get_confidence(normalized)
                hosts.append({
                    'raw': raw,
                    'normalized': self.normalizer.normalize_hostname(normalized),
                    'column': col,
                    'confidence': conf
                })
                
        for host in hosts:
            norm_host = host['normalized'].lower().strip()
            
            if len(norm_host) < 2:
                continue
                
            if norm_host not in self.discovered_hosts:
                self.discovered_hosts[norm_host] = {
                    'raw_forms': set(),
                    'occurrences': [],
                    'associated_data': defaultdict(lambda: defaultdict(set))
                }
                self.scan_metrics['hosts'] += 1
                
            self.discovered_hosts[norm_host]['raw_forms'].add(host['raw'])
            
            if len(self.discovered_hosts[norm_host]['occurrences']) < 100:
                self.discovered_hosts[norm_host]['occurrences'].append({
                    'table': table,
                    'column': host['column'],
                    'confidence': host['confidence']
                })
                
            for c, v in normalized_row.items():
                if c != host['column']:
                    if len(self.discovered_hosts[norm_host]['associated_data'][c]['raw']) < 50:
                        self.discovered_hosts[norm_host]['associated_data'][c]['raw'].add(v['raw'])
                    if len(self.discovered_hosts[norm_host]['associated_data'][c]['normalized']) < 50:
                        self.discovered_hosts[norm_host]['associated_data'][c]['normalized'].add(v['normalized'])
                        
    async def _analyze(self):
        logger.info("Phase 3: Analysis...")
        
        self.column_classifier.discover_types()
        
        total = sum(self.column_importance.values())
        self.important_columns = []
        
        for col, count in self.column_importance.items():
            importance = count / total if total > 0 else 0
            col_type = self.column_classifier.classify(col)
            
            if importance > 0.0001:
                self.important_columns.append({
                    'name': col,
                    'importance': importance,
                    'occurrences': count,
                    'type': col_type
                })
                
        self.important_columns.sort(key=lambda x: x['importance'], reverse=True)
        self.important_columns = self.important_columns[:self.config.get('column_limit', 100)]
        
        logger.info(f"Found {len(self.important_columns)} important columns")
        
    async def _build_cmdb(self):
        logger.info("Phase 4: Building CMDB...")
        
        await self.cmdb_builder.initialize()
        
        schema = ['hostname', 'raw_forms', 'occurrence_count', 'confidence']
        for col in self.important_columns[:100]:
            schema.append(col['name'])
            
        await self.cmdb_builder.create_hosts_table(schema)
        
        records = []
        for norm_host, data in self.discovered_hosts.items():
            record = {
                'hostname': norm_host,
                'raw_forms': json.dumps(list(data['raw_forms'])[:20]),
                'occurrence_count': len(data['occurrences']),
                'confidence': max([o['confidence'] for o in data['occurrences']] or [0])
            }
            
            for col in self.important_columns[:100]:
                col_name = col['name']
                if col_name in data['associated_data']:
                    vals = list(data['associated_data'][col_name]['normalized'])[:10]
                    if len(vals) == 1:
                        record[col_name] = vals[0]
                    elif len(vals) > 1:
                        record[col_name] = json.dumps(vals)
                    else:
                        record[col_name] = None
                else:
                    record[col_name] = None
                    
            records.append(record)
            
            if len(records) >= 1000:
                await self.cmdb_builder.bulk_insert('hosts', records)
                records = []
                gc.collect()
                
        if records:
            await self.cmdb_builder.bulk_insert('hosts', records)
            
        await self.cmdb_builder.create_indexes(['hostname', 'confidence', 'occurrence_count'])
        
        self.cmdb_builder.export_to_csv('cmdb_backup.csv')
        
        logger.info(f"CMDB built: {len(self.discovered_hosts)} hosts")
        
    def _report(self):
        elapsed = (datetime.now() - self.scan_metrics['start_time']).total_seconds()
        
        print(f"""
{"="*60}
CMDB+ COMPLETE
{"="*60}
Projects: {self.scan_metrics['projects']}
Datasets: {self.scan_metrics['datasets']}
Tables: {self.scan_metrics['tables']}
Tables skipped: {self.scan_metrics['tables_skipped']}
Tables failed: {self.scan_metrics['tables_failed']}
Rows: {self.scan_metrics['rows']:,}
Hosts: {len(self.discovered_hosts):,}
Errors: {self.scan_metrics['errors']}
Time: {elapsed:.1f}s
Rate: {self.scan_metrics['rows']/elapsed:.0f} rows/s
{"="*60}
""")
        
        stats_file = Path('cmdb_statistics.json')
        with open(stats_file, 'w') as f:
            json.dump({
                'metrics': self.scan_metrics,
                'hosts': len(self.discovered_hosts),
                'columns': [c['name'] for c in self.important_columns[:50]],
                'elapsed': elapsed,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2, default=str)

async def main():
    config_file = Path('config.json')
    
    if not config_file.exists():
        default = {
            'projects': ['your-project-id'],
            'max_workers': 3,
            'rows_per_batch': 5000,
            'use_daily_cache': True
        }
        with open(config_file, 'w') as f:
            json.dump(default, f, indent=2)
        print(f"Edit {config_file} with your BigQuery project IDs")
        sys.exit(1)
        
    with open(config_file, 'r') as f:
        config = json.load(f)
        
    projects = config.get('projects', [])
    
    if not projects or projects == ['your-project-id']:
        print(f"Edit {config_file} with your actual project IDs")
        sys.exit(1)
        
    orchestrator = CMDBPlusOrchestrator(projects, config)
    await orchestrator.execute()

if __name__ == "__main__":
    asyncio.run(main())