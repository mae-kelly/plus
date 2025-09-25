#!/usr/bin/env python3

import os
import sys
import json
import asyncio
import pickle
import signal
from pathlib import Path
from typing import Dict, List, Set, Any, Optional, Tuple
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import gc
from google.cloud import bigquery
from google.api_core import exceptions
import time
import hashlib
import re
import traceback
import psutil
import numpy as np

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

logging.getLogger('google.cloud').setLevel(logging.WARNING)
logging.getLogger('google.auth').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)

class CMDBPlusOrchestrator:
    
    def __init__(self, project_ids: List[str], config: Dict[str, Any] = None):
        self.project_ids = project_ids
        self.config = config or self._load_config()
        
        self.cache_dir = Path('cache')
        self.cache_dir.mkdir(exist_ok=True)
        self.checkpoints_dir = Path('checkpoints')
        self.checkpoints_dir.mkdir(exist_ok=True)
        self.outputs_dir = Path('outputs')
        self.outputs_dir.mkdir(exist_ok=True)
        
        today = datetime.now().strftime("%Y%m%d")
        self.daily_cache_file = self.cache_dir / f'cmdb_cache_{today}.pkl'
        self.training_cache_file = self.cache_dir / 'training_samples.pkl'
        self.table_schema_cache_file = self.cache_dir / 'table_schemas.pkl'
        self.checkpoint_file = self.checkpoints_dir / 'cmdb_checkpoint.pkl'
        
        self.discovered_hosts = {}
        self.column_importance = defaultdict(int)
        self.column_value_samples = defaultdict(list)
        self.processed_tables = set()
        self.failed_tables = set()
        self.skipped_tables = set()
        self.table_schemas_cache = {}
        self.query_results_cache = {}
        self.failed_queries = set()
        self.problematic_columns = defaultdict(set)
        self.complex_columns = defaultdict(set)
        
        self.scan_metrics = {
            'projects': 0,
            'datasets': 0,
            'tables': 0,
            'tables_skipped': 0,
            'tables_failed': 0,
            'rows': 0,
            'hosts': 0,
            'errors': 0,
            'columns_skipped': 0,
            'queries_cached': 0,
            'memory_cleanups': 0,
            'start_time': datetime.now()
        }
        
        self.table_filters = {
            'skip_patterns': self.config.get('skip_table_patterns', [
                r'.*_staging$',
                r'.*_temp$',
                r'.*_tmp$',
                r'.*_backup$',
                r'.*_old$',
                r'.*_test$',
                r'.*_dev$',
                r'^temp_',
                r'^tmp_',
                r'^staging_',
                r'^test_',
                r'.*_\d{8}$',
                r'.*_\d{14}$',
                r'.*_copy\d*$',
                r'.*_bak$',
            ]),
            'skip_prefixes': [
                'INFORMATION_SCHEMA',
                '__TABLES__',
                '_SESSION',
                '_SCRIPT',
                '_TEMP',
            ],
            'include_patterns': self.config.get('include_table_patterns', []),
            'focus_datasets': self.config.get('focus_datasets', []),
        }
        
        self.column_filters = {
            'skip_types': ['RECORD', 'STRUCT', 'ARRAY', 'GEOGRAPHY', 'JSON', 'BYTES', 'REPEATED', 'NESTED'],
            'skip_names': ['__root__', '__key__', '__error__', '__null__'],
            'hostname_patterns': [
                r'.*host.*',
                r'.*server.*',
                r'.*machine.*',
                r'.*node.*',
                r'.*instance.*',
                r'.*computer.*',
                r'.*device.*',
                r'.*endpoint.*',
                r'.*asset.*',
                r'.*workstation.*',
                r'.*desktop.*',
                r'.*laptop.*',
                r'.*ip.*address.*',
                r'.*fqdn.*',
                r'.*dns.*',
                r'.*domain.*name.*',
            ]
        }
        
        self.query_timeout = self.config.get('query_timeout', 30)
        self.max_retries = self.config.get('max_retries', 3)
        self.batch_size = self.config.get('rows_per_batch', 10000)
        self.max_memory_percent = self.config.get('max_memory_percent', 80)
        self.checkpoint_interval = self.config.get('checkpoint_interval', 1000)
        
        self.partition_lookback_days = 30
        
        self.shutdown_requested = False
        self.using_cache = False
        self.executor = None
        self.important_columns = []
        
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self._initialize_gpu()
        self._initialize_components()
        
        max_workers = self.config.get('max_workers', 5)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        self._load_cache_or_checkpoint()
        
    def _load_config(self) -> Dict[str, Any]:
        config_file = Path('config.json')
        
        default_config = {
            'projects': ['your-project-id'],
            'max_workers': 5,
            'max_memory_mb': 8192,
            'rows_per_batch': 10000,
            'checkpoint_enabled': True,
            'checkpoint_interval': 1000,
            'use_daily_cache': True,
            'cache_expiry_hours': 24,
            'process_all_data': True,
            'smart_table_filtering': True,
            'parallel_processing': True,
            'skip_failed_tables': True,
            'retry_failed_queries': False,
            'max_training_time_seconds': 180,
            'max_training_samples': 50000,
            'training_sample_per_table': 1000,
            'column_limit': 100,
            'max_raw_forms_per_host': 20,
            'max_occurrences_per_host': 100,
            'max_values_per_column': 50,
            'query_timeout': 30,
            'max_retries': 3,
            'max_memory_percent': 80,
            'skip_table_patterns': [],
            'include_table_patterns': [],
            'focus_datasets': [],
        }
        
        if config_file.exists():
            with open(config_file, 'r') as f:
                loaded_config = json.load(f)
                default_config.update(loaded_config)
                return default_config
        else:
            with open(config_file, 'w') as f:
                json.dump(default_config, f, indent=2)
            logger.info(f"Created config.json - please edit with your project IDs")
            return default_config
            
    def _signal_handler(self, signum, frame):
        logger.info("Shutdown signal received. Saving state...")
        self.shutdown_requested = True
        self._save_checkpoint()
        if self.executor:
            self.executor.shutdown(wait=False)
        sys.exit(0)
        
    def _initialize_gpu(self):
        from gpu_accelerator import GPUAccelerator
        self.gpu = GPUAccelerator()
        self.device = self.gpu.initialize()
        
        if 'cpu' in str(self.device).lower():
            logger.error("GPU required. Mac M1/M2/M3 with MPS support required.")
            sys.exit(1)
            
        logger.info(f"GPU initialized: {self.device}")
        
    def _initialize_components(self):
        logger.info("Initializing CMDB+ components...")
        
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
        
        logger.info("All components initialized successfully")
        
    def _check_memory(self) -> bool:
        memory = psutil.virtual_memory()
        percent_used = memory.percent
        
        if percent_used > self.max_memory_percent:
            logger.warning(f"Memory usage at {percent_used}% - triggering cleanup")
            gc.collect()
            self.scan_metrics['memory_cleanups'] += 1
            
            memory = psutil.virtual_memory()
            if memory.percent > self.max_memory_percent:
                logger.error("Memory still high after cleanup")
                return False
                
        return True
        
    def _should_skip_table(self, table_id: str, dataset_id: str = None) -> bool:
        if not self.config.get('smart_table_filtering', True):
            return False
            
        if self.table_filters['focus_datasets'] and dataset_id:
            if dataset_id not in self.table_filters['focus_datasets']:
                return True
                
        for prefix in self.table_filters['skip_prefixes']:
            if table_id.startswith(prefix):
                logger.debug(f"Skipping {table_id} - matches skip prefix")
                return True
                
        for pattern in self.table_filters['skip_patterns']:
            if re.match(pattern, table_id, re.IGNORECASE):
                logger.debug(f"Skipping {table_id} - matches skip pattern")
                return True
                
        if self.table_filters['include_patterns']:
            for pattern in self.table_filters['include_patterns']:
                if re.match(pattern, table_id, re.IGNORECASE):
                    return False
            logger.debug(f"Skipping {table_id} - doesn't match include patterns")
            return True
            
        return False
        
    def _load_cache_or_checkpoint(self):
        if self.daily_cache_file.exists() and self.config.get('use_daily_cache', True):
            try:
                with open(self.daily_cache_file, 'rb') as f:
                    cache = pickle.load(f)
                    
                cache_time = cache.get('timestamp', datetime.min)
                cache_age_hours = (datetime.now() - cache_time).total_seconds() / 3600
                
                if cache_age_hours < self.config.get('cache_expiry_hours', 24):
                    self.discovered_hosts = cache.get('discovered_hosts', {})
                    self.column_importance = cache.get('column_importance', defaultdict(int))
                    self.processed_tables = cache.get('processed_tables', set())
                    self.failed_tables = cache.get('failed_tables', set())
                    self.column_value_samples = cache.get('column_value_samples', defaultdict(list))
                    self.scan_metrics = cache.get('scan_metrics', self.scan_metrics)
                    
                    logger.info(f"Loaded cache from {cache_time.strftime('%Y-%m-%d %H:%M')}")
                    logger.info(f"Cache: {len(self.discovered_hosts)} hosts, {len(self.processed_tables)} tables")
                    
                    if len(self.processed_tables) > 0:
                        print(f"\nFound cached data from {cache_time.strftime('%H:%M today')}")
                        print(f"Cache contains {len(self.discovered_hosts)} hosts from {len(self.processed_tables)} tables")
                        response = input("Use cached data? (y/n, default=y): ").strip().lower()
                        
                        if response != 'n':
                            self.using_cache = True
                            return
                            
            except Exception as e:
                logger.debug(f"Cache load failed: {e}")
                
        if self.checkpoint_file.exists() and self.config.get('checkpoint_enabled', True):
            try:
                with open(self.checkpoint_file, 'rb') as f:
                    checkpoint = pickle.load(f)
                    
                self.discovered_hosts = checkpoint.get('discovered_hosts', {})
                self.column_importance = checkpoint.get('column_importance', defaultdict(int))
                self.processed_tables = checkpoint.get('processed_tables', set())
                self.failed_tables = checkpoint.get('failed_tables', set())
                self.column_value_samples = checkpoint.get('column_value_samples', defaultdict(list))
                self.scan_metrics = checkpoint.get('scan_metrics', self.scan_metrics)
                
                logger.info(f"Resumed from checkpoint: {len(self.processed_tables)} tables processed")
                
            except Exception as e:
                logger.debug(f"Checkpoint load failed: {e}")
                
        self.using_cache = False
        
    def _save_checkpoint(self):
        if not self.config.get('checkpoint_enabled', True):
            return
            
        checkpoint = {
            'discovered_hosts': self.discovered_hosts,
            'column_importance': dict(self.column_importance),
            'processed_tables': self.processed_tables,
            'failed_tables': self.failed_tables,
            'column_value_samples': dict(self.column_value_samples),
            'scan_metrics': self.scan_metrics,
            'timestamp': datetime.now()
        }
        
        try:
            temp_file = self.checkpoint_file.with_suffix('.tmp')
            with open(temp_file, 'wb') as f:
                pickle.dump(checkpoint, f, protocol=pickle.HIGHEST_PROTOCOL)
            temp_file.replace(self.checkpoint_file)
            logger.debug("Checkpoint saved")
        except Exception as e:
            logger.error(f"Checkpoint save failed: {e}")
            
    def _save_daily_cache(self):
        if not self.config.get('use_daily_cache', True):
            return
            
        cache = {
            'discovered_hosts': self.discovered_hosts,
            'column_importance': dict(self.column_importance),
            'processed_tables': self.processed_tables,
            'failed_tables': self.failed_tables,
            'column_value_samples': dict(self.column_value_samples),
            'scan_metrics': self.scan_metrics,
            'timestamp': datetime.now()
        }
        
        try:
            with open(self.daily_cache_file, 'wb') as f:
                pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info("Daily cache saved")
            
            for cache_file in self.cache_dir.glob('cmdb_cache_*.pkl'):
                age_days = (datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)).days
                if age_days > 7:
                    cache_file.unlink()
                    
        except Exception as e:
            logger.error(f"Cache save failed: {e}")
            
    def _get_safe_columns(self, schema) -> List[str]:
        safe_columns = []
        
        for field in schema:
            if self.config.get('skip_complex_columns', True):
                if field.field_type in self.column_filters['skip_types']:
                    logger.debug(f"Skipping complex column {field.name} (type: {field.field_type})")
                    self.scan_metrics['columns_skipped'] += 1
                    continue
                    
            if field.name in self.column_filters['skip_names']:
                continue
                
            safe_columns.append(field.name)
            
        return safe_columns
        
    def _safe_query(self, client: bigquery.Client, query: str, 
                   timeout: int = None) -> Optional[List]:
        timeout = timeout or self.query_timeout
        
        query_hash = hashlib.md5(query.encode()).hexdigest()[:8]
        
        if query_hash in self.failed_queries and not self.config.get('retry_failed_queries', False):
            return None
            
        if query_hash in self.query_results_cache:
            self.scan_metrics['queries_cached'] += 1
            return self.query_results_cache[query_hash]
            
        job_config = bigquery.QueryJobConfig()
        job_config.use_query_cache = True
        job_config.use_legacy_sql = False
        
        for attempt in range(self.max_retries):
            try:
                query_job = client.query(query, job_config=job_config)
                results = list(query_job.result(timeout=timeout))
                
                if len(results) < 10000:
                    self.query_results_cache[query_hash] = results
                    
                return results
                
            except exceptions.BadRequest as e:
                error_msg = str(e)
                
                if 'without a filter over' in error_msg.lower() or 'requires a partition filter' in error_msg.lower():
                    if '_PARTITIONTIME' in error_msg and 'WHERE' not in query.upper():
                        modified_query = query.replace(
                            'LIMIT',
                            f'WHERE DATE(_PARTITIONTIME) >= DATE_SUB(CURRENT_DATE(), INTERVAL {self.partition_lookback_days} DAY) LIMIT'
                        )
                        return self._safe_query(client, modified_query, timeout)
                    elif 'WHERE' not in query.upper():
                        modified_query = query.replace('LIMIT', 'WHERE TRUE LIMIT')
                        return self._safe_query(client, modified_query, timeout)
                        
                if 'Unrecognized name' in error_msg:
                    match = re.search(r'Unrecognized name: ([^\s]+)', error_msg)
                    if match:
                        bad_column = match.group(1).strip('`')
                        logger.debug(f"Column {bad_column} not found")
                        
                logger.debug(f"Query failed: {error_msg[:200]}")
                self.failed_queries.add(query_hash)
                return None
                
            except exceptions.Forbidden:
                logger.debug("Permission denied")
                self.failed_queries.add(query_hash)
                return None
                
            except exceptions.NotFound:
                logger.debug("Table/dataset not found")
                self.failed_queries.add(query_hash)
                return None
                
            except exceptions.DeadlineExceeded:
                logger.debug("Query timeout")
                if attempt < self.max_retries - 1:
                    timeout = timeout * 2
                    time.sleep(1)
                    continue
                self.failed_queries.add(query_hash)
                return None
                
            except Exception as e:
                logger.debug(f"Query error: {str(e)[:100]}")
                if attempt < self.max_retries - 1:
                    time.sleep(1)
                    continue
                self.failed_queries.add(query_hash)
                return None
                
        return None
        
    async def execute(self):
        logger.info(f"Starting CMDB+ for projects: {self.project_ids}")
        
        if self.using_cache and len(self.discovered_hosts) > 0:
            logger.info("Using cached data, skipping to CMDB build")
            await self._build_cmdb()
            self._report_statistics()
            return
            
        try:
            await self._learn_patterns()
            await self._discover_all_data()
            await self._analyze_and_aggregate()
            
            self._save_daily_cache()
            
            await self._build_cmdb()
            self._report_statistics()
            
            if self.checkpoint_file.exists():
                self.checkpoint_file.unlink()
                logger.info("Checkpoint file removed after successful completion")
                
        except Exception as e:
            logger.error(f"Execution failed: {e}", exc_info=True)
            self._save_checkpoint()
            raise
            
        finally:
            if self.executor:
                self.executor.shutdown(wait=True)
                
    async def _learn_patterns(self):
        logger.info("Phase 1: Learning hostname patterns...")
        
        if self.training_cache_file.exists():
            try:
                with open(self.training_cache_file, 'rb') as f:
                    training_samples = pickle.load(f)
                    logger.info(f"Loaded {len(training_samples)} cached training samples")
                    
                    if training_samples:
                        await self.discovery_engine.train(training_samples)
                        await self.pattern_model.learn_patterns(training_samples)
                        return
            except:
                pass
                
        training_samples = []
        start_time = time.time()
        max_time = self.config.get('max_training_time_seconds', 180)
        max_samples = self.config.get('max_training_samples', 50000)
        
        for project_id in self.project_ids:
            if len(training_samples) >= max_samples or (time.time() - start_time) > max_time:
                break
                
            try:
                client = self.bq_manager.get_client(project_id)
                datasets = list(client.list_datasets(max_results=10))
                
                for dataset_ref in datasets[:5]:
                    if len(training_samples) >= max_samples:
                        break
                        
                    dataset_id = dataset_ref.dataset_id
                    
                    if self.table_filters['focus_datasets'] and dataset_id not in self.table_filters['focus_datasets']:
                        continue
                        
                    try:
                        tables = list(client.list_tables(dataset_ref, max_results=20))
                        
                        for table_ref in tables[:10]:
                            if len(training_samples) >= max_samples:
                                break
                                
                            table_id = table_ref.table_id
                            
                            if self._should_skip_table(table_id, dataset_id):
                                continue
                                
                            full_table_name = f"{project_id}.{dataset_id}.{table_id}"
                            
                            if full_table_name in self.failed_tables:
                                continue
                                
                            try:
                                table = client.get_table(table_ref)
                                
                                hostname_columns = []
                                for field in table.schema:
                                    if field.field_type in self.column_filters['skip_types']:
                                        continue
                                        
                                    field_lower = field.name.lower()
                                    for pattern in self.column_filters['hostname_patterns']:
                                        if re.match(pattern, field_lower):
                                            hostname_columns.append(field.name)
                                            break
                                            
                                if hostname_columns:
                                    for col_name in hostname_columns[:3]:
                                        query = f"""
                                        SELECT DISTINCT `{col_name}` as value
                                        FROM `{project_id}.{dataset_id}.{table_id}`
                                        WHERE `{col_name}` IS NOT NULL
                                        LIMIT {self.config.get('training_sample_per_table', 1000)}
                                        """
                                        
                                        results = self._safe_query(client, query, timeout=10)
                                        
                                        if results:
                                            for row in results:
                                                if hasattr(row, 'value') and row.value:
                                                    training_samples.append(str(row.value))
                                                    
                            except Exception as e:
                                logger.debug(f"Training table skip {table_id}: {e}")
                                
                    except Exception as e:
                        logger.debug(f"Training dataset skip {dataset_id}: {e}")
                        
            except Exception as e:
                logger.error(f"Training project error {project_id}: {e}")
                
        if training_samples:
            try:
                with open(self.training_cache_file, 'wb') as f:
                    pickle.dump(training_samples[:max_samples], f)
            except:
                pass
                
            logger.info(f"Training on {len(training_samples)} samples")
            await self.discovery_engine.train(training_samples[:max_samples])
            await self.pattern_model.learn_patterns(training_samples[:max_samples])
        else:
            logger.warning("No training samples found, using heuristics only")
            
        logger.info(f"Training complete in {time.time()-start_time:.1f} seconds")
        
    async def _discover_all_data(self):
        logger.info("Phase 2: Discovering all hosts and data...")
        
        for project_id in self.project_ids:
            if self.shutdown_requested:
                break
                
            await self._process_project(project_id)
            
    async def _process_project(self, project_id: str):
        try:
            client = self.bq_manager.get_client(project_id)
            self.scan_metrics['projects'] += 1
            
            datasets = list(client.list_datasets())
            logger.info(f"Processing {len(datasets)} datasets in {project_id}")
            
            for dataset_ref in datasets:
                if self.shutdown_requested:
                    break
                    
                await self._process_dataset(client, project_id, dataset_ref)
                
        except Exception as e:
            logger.error(f"Failed to process project {project_id}: {e}")
            self.scan_metrics['errors'] += 1
            
    async def _process_dataset(self, client, project_id: str, dataset_ref):
        dataset_id = dataset_ref.dataset_id
        self.scan_metrics['datasets'] += 1
        
        if self.table_filters['focus_datasets'] and dataset_id not in self.table_filters['focus_datasets']:
            logger.debug(f"Skipping dataset {dataset_id} - not in focus list")
            return
            
        try:
            tables = list(client.list_tables(dataset_ref))
            logger.info(f"Processing {len(tables)} tables in {dataset_id}")
            
            table_futures = []
            
            for table_ref in tables:
                if self.shutdown_requested:
                    break
                    
                table_id = table_ref.table_id
                full_table_name = f"{project_id}.{dataset_id}.{table_id}"
                
                if full_table_name in self.processed_tables:
                    continue
                    
                if full_table_name in self.failed_tables and self.config.get('skip_failed_tables', True):
                    continue
                    
                if self._should_skip_table(table_id, dataset_id):
                    self.skipped_tables.add(full_table_name)
                    self.scan_metrics['tables_skipped'] += 1
                    continue
                    
                if self.config.get('parallel_processing', True):
                    future = self.executor.submit(
                        self._process_table,
                        client, project_id, dataset_id, table_id, full_table_name
                    )
                    table_futures.append(future)
                    
                    if len(table_futures) >= self.config.get('max_workers', 5):
                        for future in as_completed(table_futures):
                            try:
                                future.result()
                            except Exception as e:
                                logger.error(f"Table processing failed: {e}")
                                self.scan_metrics['errors'] += 1
                        table_futures = []
                else:
                    self._process_table(client, project_id, dataset_id, table_id, full_table_name)
                    
            for future in as_completed(table_futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Table processing failed: {e}")
                    self.scan_metrics['errors'] += 1
                    
        except Exception as e:
            logger.error(f"Failed to process dataset {dataset_id}: {e}")
            self.scan_metrics['errors'] += 1
            
    def _process_table(self, client, project_id: str, dataset_id: str, 
                      table_id: str, full_table_name: str):
        try:
            self.scan_metrics['tables'] += 1
            
            table = client.get_table(f"{project_id}.{dataset_id}.{table_id}")
            total_rows = table.num_rows or 0
            
            if total_rows == 0:
                logger.debug(f"Table {table_id} is empty")
                self.processed_tables.add(full_table_name)
                return
                
            logger.info(f"Processing table {self.scan_metrics['tables']}: {full_table_name} ({total_rows:,} rows)")
            
            safe_columns = self._get_safe_columns(table.schema)
            
            if not safe_columns:
                logger.warning(f"No safe columns in {table_id}, trying SELECT *")
                safe_columns = None
                
            processed_rows = 0
            consecutive_failures = 0
            max_consecutive_failures = 3
            
            while processed_rows < total_rows and consecutive_failures < max_consecutive_failures:
                if self.shutdown_requested:
                    break
                    
                if not self._check_memory():
                    logger.warning("Memory limit reached, saving checkpoint")
                    self._save_checkpoint()
                    break
                    
                if safe_columns:
                    escaped_columns = [f"`{col}`" for col in safe_columns]
                    columns_str = ', '.join(escaped_columns)
                    query = f"""
                    SELECT {columns_str}
                    FROM `{project_id}.{dataset_id}.{table_id}`
                    LIMIT {self.batch_size}
                    OFFSET {processed_rows}
                    """
                else:
                    query = f"""
                    SELECT *
                    FROM `{project_id}.{dataset_id}.{table_id}`
                    LIMIT {self.batch_size}
                    OFFSET {processed_rows}
                    """
                    
                results = self._safe_query(client, query, timeout=60)
                
                if results is None:
                    consecutive_failures += 1
                    logger.warning(f"Query failed for {table_id} (attempt {consecutive_failures}/{max_consecutive_failures})")
                    
                    if consecutive_failures >= max_consecutive_failures:
                        logger.error(f"Giving up on table {table_id} after {max_consecutive_failures} failures")
                        self.failed_tables.add(full_table_name)
                        self.scan_metrics['tables_failed'] += 1
                        break
                        
                    self.batch_size = max(1000, self.batch_size // 2)
                    time.sleep(1)
                    continue
                    
                consecutive_failures = 0
                batch_count = 0
                
                for row in results:
                    self.scan_metrics['rows'] += 1
                    batch_count += 1
                    
                    self._process_row(dict(row), full_table_name, table.schema)
                    
                    if self.scan_metrics['rows'] % self.checkpoint_interval == 0:
                        self._save_checkpoint()
                        
                processed_rows += batch_count
                
                if batch_count == 0:
                    break
                    
                if batch_count < self.batch_size:
                    break
                    
                logger.debug(f"Processed {processed_rows:,}/{total_rows:,} rows from {table_id}")
                
            self.processed_tables.add(full_table_name)
            
            if safe_columns and len(safe_columns) < len(table.schema):
                skipped = len(table.schema) - len(safe_columns)
                logger.info(f"Completed {full_table_name}: {processed_rows:,} rows ({skipped} columns skipped)")
            else:
                logger.info(f"Completed {full_table_name}: {processed_rows:,} rows")
                
        except Exception as e:
            logger.error(f"Failed to process table {full_table_name}: {e}")
            self.failed_tables.add(full_table_name)
            self.scan_metrics['tables_failed'] += 1
            self.scan_metrics['errors'] += 1
            
    def _process_row(self, row: Dict[str, Any], table_name: str, schema):
        potential_hosts = []
        row_normalized = {}
        
        for column, value in row.items():
            if value is None:
                continue
                
            self.column_importance[column] += 1
            
            if len(self.column_value_samples[column]) < 100:
                self.column_value_samples[column].append(value)
                
            self.column_classifier.observe(column, value, table_name)
            
            normalized = self.normalizer.normalize(value)
            raw_value = str(value)
            
            row_normalized[column] = {'raw': raw_value, 'normalized': normalized}
            
            if self.discovery_engine.is_hostname(normalized):
                confidence = self.discovery_engine.get_confidence(normalized)
                
                if self.pattern_model.predict(normalized) > 0.5:
                    confidence = max(confidence, self.pattern_model.predict(normalized))
                    
                potential_hosts.append({
                    'raw': raw_value,
                    'normalized': self.normalizer.normalize_hostname(normalized),
                    'column': column,
                    'confidence': confidence
                })
                
        for host_info in potential_hosts:
            normalized_host = host_info['normalized'].lower().strip()
            
            if not normalized_host or len(normalized_host) < 2:
                continue
                
            if normalized_host not in self.discovered_hosts:
                self.discovered_hosts[normalized_host] = {
                    'raw_forms': set(),
                    'occurrences': [],
                    'associated_data': defaultdict(lambda: defaultdict(set))
                }
                self.scan_metrics['hosts'] += 1
                
            self.discovered_hosts[normalized_host]['raw_forms'].add(host_info['raw'])
            
            max_occurrences = self.config.get('max_occurrences_per_host', 100)
            if len(self.discovered_hosts[normalized_host]['occurrences']) < max_occurrences:
                self.discovered_hosts[normalized_host]['occurrences'].append({
                    'table': table_name,
                    'column': host_info['column'],
                    'confidence': host_info['confidence']
                })
                
            max_values = self.config.get('max_values_per_column', 50)
            for col, values in row_normalized.items():
                if col != host_info['column']:
                    if len(self.discovered_hosts[normalized_host]['associated_data'][col]['raw']) < max_values:
                        self.discovered_hosts[normalized_host]['associated_data'][col]['raw'].add(values['raw'])
                    if len(self.discovered_hosts[normalized_host]['associated_data'][col]['normalized']) < max_values:
                        self.discovered_hosts[normalized_host]['associated_data'][col]['normalized'].add(values['normalized'])
                        
    async def _analyze_and_aggregate(self):
        logger.info("Phase 3: Analyzing and aggregating...")
        
        self.column_classifier.discover_types()
        
        total_occurrences = sum(self.column_importance.values())
        
        self.important_columns = []
        for column, count in self.column_importance.items():
            importance = count / total_occurrences if total_occurrences > 0 else 0
            
            col_type = self.column_classifier.classify(column)
            
            if importance > 0.0001:
                self.important_columns.append({
                    'name': column,
                    'importance': importance,
                    'occurrences': count,
                    'type': col_type
                })
                
        self.important_columns.sort(key=lambda x: x['importance'], reverse=True)
        
        column_limit = self.config.get('column_limit', 100)
        self.important_columns = self.important_columns[:column_limit]
        
        logger.info(f"Identified {len(self.important_columns)} important columns")
        logger.info(f"Top 20 columns: {[c['name'] for c in self.important_columns[:20]]}")
        
        relationships = self.relationship_analyzer.analyze_host_relationships(self.discovered_hosts)
        logger.info(f"Found {len(relationships.get('clusters', []))} host clusters")
        
    async def _build_cmdb(self):
        logger.info("Phase 4: Building CMDB...")
        
        await self.cmdb_builder.initialize()
        
        schema_columns = ['hostname', 'raw_forms', 'occurrence_count', 'confidence']
        
        column_limit = self.config.get('column_limit', 100)
        for col_info in self.important_columns[:column_limit]:
            schema_columns.append(col_info['name'])
            
        await self.cmdb_builder.create_hosts_table(schema_columns)
        
        host_records = []
        batch_size = 1000
        max_raw_forms = self.config.get('max_raw_forms_per_host', 20)
        
        for i, (normalized_host, data) in enumerate(self.discovered_hosts.items()):
            record = {
                'hostname': normalized_host,
                'raw_forms': json.dumps(list(data['raw_forms'])[:max_raw_forms]),
                'occurrence_count': len(data['occurrences']),
                'confidence': max([o['confidence'] for o in data['occurrences']] or [0])
            }
            
            for col_info in self.important_columns[:column_limit]:
                col_name = col_info['name']
                
                if col_name in data['associated_data']:
                    normalized_values = list(data['associated_data'][col_name]['normalized'])[:10]
                    
                    if len(normalized_values) == 1:
                        record[col_name] = normalized_values[0]
                    elif len(normalized_values) > 1:
                        record[col_name] = json.dumps(normalized_values)
                    else:
                        record[col_name] = None
                else:
                    record[col_name] = None
                    
            host_records.append(record)
            
            if len(host_records) >= batch_size:
                await self.cmdb_builder.bulk_insert('hosts', host_records)
                host_records = []
                gc.collect()
                
        if host_records:
            await self.cmdb_builder.bulk_insert('hosts', host_records)
            
        index_columns = ['hostname', 'occurrence_count', 'confidence']
        index_columns.extend([c['name'] for c in self.important_columns[:20]])
        
        await self.cmdb_builder.create_indexes(index_columns)
        
        self.cmdb_builder.export_to_csv(self.outputs_dir / 'cmdb_export.csv')
        
        logger.info(f"CMDB built with {len(self.discovered_hosts)} unique hosts")
        
    def _report_statistics(self):
        elapsed = (datetime.now() - self.scan_metrics['start_time']).total_seconds()
        
        cache_status = 'Used cached data' if self.using_cache else 'Fresh scan'
        
        stats_report = f"""
{"="*60}
CMDB+ DISCOVERY COMPLETE
{"="*60}
Projects scanned: {self.scan_metrics['projects']}
Datasets scanned: {self.scan_metrics['datasets']}
Tables scanned: {self.scan_metrics['tables']}
Tables skipped: {self.scan_metrics['tables_skipped']}
Tables failed: {self.scan_metrics['tables_failed']}
Rows processed: {self.scan_metrics['rows']:,}
Unique hosts discovered: {len(self.discovered_hosts):,}
Important columns identified: {len(self.important_columns)}
Columns skipped: {self.scan_metrics['columns_skipped']}
Queries cached: {self.scan_metrics['queries_cached']}
Memory cleanups: {self.scan_metrics['memory_cleanups']}
Errors encountered: {self.scan_metrics['errors']}
Time elapsed: {elapsed:.1f} seconds
Processing rate: {self.scan_metrics['rows']/elapsed:.0f} rows/second
Cache status: {cache_status}
{"="*60}
"""
        logger.info(stats_report)
        print(stats_report)
        
        stats_file = self.outputs_dir / 'cmdb_statistics.json'
        with open(stats_file, 'w') as f:
            json.dump({
                'scan_metrics': self.scan_metrics,
                'hosts_discovered': len(self.discovered_hosts),
                'tables_processed': len(self.processed_tables),
                'tables_failed': len(self.failed_tables),
                'tables_skipped': len(self.skipped_tables),
                'important_columns': [c['name'] for c in self.important_columns[:50]],
                'elapsed_seconds': elapsed,
                'cache_used': self.using_cache,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2, default=str)

async def main():
    config_file = Path('config.json')
    
    if not config_file.exists():
        default_config = {
            'projects': ['your-project-id-1', 'your-project-id-2'],
            'max_workers': 5,
            'rows_per_batch': 10000,
            'use_daily_cache': True,
            'process_all_data': True,
            'smart_table_filtering': True,
            'column_limit': 100
        }
        
        with open(config_file, 'w') as f:
            json.dump(default_config, f, indent=2)
            
        logger.error(f"Please edit {config_file} with your BigQuery project IDs and run again")
        sys.exit(1)
        
    with open(config_file, 'r') as f:
        config = json.load(f)
        
    projects = config.get('projects', [])
    
    if not projects or 'your-project-id' in str(projects):
        logger.error(f"Please edit {config_file} with your actual BigQuery project IDs")
        sys.exit(1)
        
    orchestrator = CMDBPlusOrchestrator(projects, config)
    await orchestrator.execute()

if __name__ == "__main__":
    asyncio.run(main())