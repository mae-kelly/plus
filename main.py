import os
import sys
import json
import asyncio
import pickle
import signal
from pathlib import Path
from typing import Dict, List, Set, Any, Optional
from collections import defaultdict
from datetime import datetime, timedelta
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
from google.cloud import bigquery
from google.api_core import exceptions
import time
import hashlib

# Setup logging to file and console
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

class CMDBPlusOrchestrator:
    def __init__(self, project_ids: List[str], config: Dict[str, Any] = None):
        self.project_ids = project_ids
        self.config = config or self._load_config()
        
        # Cache directory for daily runs
        self.cache_dir = Path('cache')
        self.cache_dir.mkdir(exist_ok=True)
        
        # Today's cache file
        today = datetime.now().strftime("%Y%m%d")
        self.daily_cache_file = self.cache_dir / f'cmdb_cache_{today}.pkl'
        
        # Checkpoint file for resuming interrupted runs
        self.checkpoint_file = Path('cmdb_checkpoint.pkl')
        self.checkpoint_interval = 1000  # Save every 1000 rows
        
        # Track dataset locations for multi-region support
        self.dataset_locations = {}
        
        # Track failed queries to retry differently
        self.failed_queries = set()
        
        # Force GPU
        from gpu_accelerator import GPUAccelerator
        self.gpu = GPUAccelerator()
        self.device = self.gpu.initialize()
        if 'cpu' in str(self.device).lower():
            raise RuntimeError("GPU required. No CPU fallback.")
            
        self._initialize_components()
        
        # Load daily cache or checkpoint
        self._load_cache_or_checkpoint()
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=self.config.get('max_workers', 5))
        
        # Memory management
        self.max_memory_mb = self.config.get('max_memory_mb', 8192)
        self.rows_per_batch = self.config.get('rows_per_batch', 10000)
        
        # Signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        self.shutdown_requested = False
        
    def _load_config(self) -> Dict[str, Any]:
        config_file = Path('config.json')
        if config_file.exists():
            with open(config_file, 'r') as f:
                return json.load(f)
        else:
            # Default configuration
            default_config = {
                'max_workers': 5,
                'max_memory_mb': 8192,
                'rows_per_batch': 10000,
                'checkpoint_enabled': True,
                'retry_attempts': 3,
                'retry_delay_seconds': 5,
                'full_scan_large_tables': True,  # Process ALL rows from large tables
                'large_table_threshold_rows': 1000000,
                'use_daily_cache': True,  # Cache results for 24 hours
                'cache_expiry_hours': 24,
                'max_training_samples': 50000,
                'training_sample_per_table': 1000,
                'column_limit': 100,
                'max_raw_forms_per_host': 20,
                'max_occurrences_per_host': 100,
                'max_values_per_column': 50
            }
            with open(config_file, 'w') as f:
                json.dump(default_config, f, indent=2)
            logger.info(f"Created default config file: {config_file}")
            return default_config
            
    def _signal_handler(self, signum, frame):
        logger.info("Shutdown signal received. Saving checkpoint...")
        self.shutdown_requested = True
        self._save_checkpoint()
        sys.exit(0)
        
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
        
        logger.info("All components initialized")
        
    def _load_cache_or_checkpoint(self):
        """Load from daily cache if available and fresh, otherwise checkpoint"""
        
        # Check for today's cache first
        if self.daily_cache_file.exists() and self.config.get('use_daily_cache', True):
            try:
                with open(self.daily_cache_file, 'rb') as f:
                    cache = pickle.load(f)
                    
                cache_time = cache.get('timestamp', datetime.min)
                cache_age_hours = (datetime.now() - cache_time).total_seconds() / 3600
                
                if cache_age_hours < self.config.get('cache_expiry_hours', 24):
                    # Cache is fresh, use it
                    self.discovered_hosts = cache.get('discovered_hosts', {})
                    self.column_importance = cache.get('column_importance', defaultdict(int))
                    self.scan_metrics = cache.get('scan_metrics', {
                        'projects': 0, 'datasets': 0, 'tables': 0, 'rows': 0, 
                        'hosts': 0, 'start_time': datetime.now(), 'errors': 0
                    })
                    self.processed_tables = cache.get('processed_tables', set())
                    self.dataset_locations = cache.get('dataset_locations', {})
                    
                    logger.info(f"Loaded fresh cache from {cache_time.strftime('%Y-%m-%d %H:%M')}")
                    logger.info(f"Cache contains: {len(self.discovered_hosts)} hosts, {len(self.processed_tables)} tables")
                    
                    # Ask user if they want to use cache or rescan
                    if len(self.processed_tables) > 0:
                        print(f"\nFound cached data from {cache_time.strftime('%Y-%m-%d %H:%M')}")
                        print(f"Cache contains {len(self.discovered_hosts)} hosts from {len(self.processed_tables)} tables")
                        response = input("Use cached data? (y/n, default=y): ").strip().lower()
                        
                        if response == 'n':
                            logger.info("User chose to rescan, clearing cache")
                            self._init_tracking()
                        else:
                            logger.info("Using cached data")
                            self.using_cache = True
                            return
                else:
                    logger.info(f"Cache is {cache_age_hours:.1f} hours old, considered stale")
                    
            except Exception as e:
                logger.warning(f"Failed to load daily cache: {e}")
        
        # No valid cache, check for checkpoint from interrupted run
        if self.checkpoint_file.exists() and self.config.get('checkpoint_enabled', True):
            try:
                with open(self.checkpoint_file, 'rb') as f:
                    checkpoint = pickle.load(f)
                    
                self.discovered_hosts = checkpoint.get('discovered_hosts', {})
                self.column_importance = checkpoint.get('column_importance', defaultdict(int))
                self.scan_metrics = checkpoint.get('scan_metrics', {
                    'projects': 0, 'datasets': 0, 'tables': 0, 'rows': 0, 
                    'hosts': 0, 'start_time': datetime.now(), 'errors': 0
                })
                self.processed_tables = checkpoint.get('processed_tables', set())
                self.dataset_locations = checkpoint.get('dataset_locations', {})
                
                logger.info(f"Checkpoint loaded: {len(self.discovered_hosts)} hosts, {len(self.processed_tables)} tables processed")
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}")
                self._init_tracking()
        else:
            self._init_tracking()
            
        self.using_cache = False
            
    def _init_tracking(self):
        self.discovered_hosts = {}
        self.column_importance = defaultdict(int)
        self.scan_metrics = {
            'projects': 0, 'datasets': 0, 'tables': 0, 'rows': 0, 
            'hosts': 0, 'start_time': datetime.now(), 'errors': 0
        }
        self.processed_tables = set()
        self.dataset_locations = {}
        self.using_cache = False
        
    def _save_checkpoint(self):
        """Save current progress as checkpoint"""
        if not self.config.get('checkpoint_enabled', True):
            return
            
        checkpoint = {
            'discovered_hosts': self.discovered_hosts,
            'column_importance': dict(self.column_importance),
            'scan_metrics': self.scan_metrics,
            'processed_tables': self.processed_tables,
            'dataset_locations': self.dataset_locations,
            'timestamp': datetime.now()
        }
        
        temp_file = self.checkpoint_file.with_suffix('.tmp')
        try:
            with open(temp_file, 'wb') as f:
                pickle.dump(checkpoint, f, protocol=pickle.HIGHEST_PROTOCOL)
            temp_file.replace(self.checkpoint_file)
            logger.debug(f"Checkpoint saved: {len(self.discovered_hosts)} hosts")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            
    def _save_daily_cache(self):
        """Save results as daily cache"""
        if not self.config.get('use_daily_cache', True):
            return
            
        cache = {
            'discovered_hosts': self.discovered_hosts,
            'column_importance': dict(self.column_importance),
            'scan_metrics': self.scan_metrics,
            'processed_tables': self.processed_tables,
            'dataset_locations': self.dataset_locations,
            'timestamp': datetime.now()
        }
        
        try:
            with open(self.daily_cache_file, 'wb') as f:
                pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"Daily cache saved to {self.daily_cache_file}")
            
            # Clean up old cache files
            self._cleanup_old_caches()
        except Exception as e:
            logger.error(f"Failed to save daily cache: {e}")
            
    def _cleanup_old_caches(self):
        """Remove cache files older than 7 days"""
        try:
            for cache_file in self.cache_dir.glob('cmdb_cache_*.pkl'):
                file_age_days = (datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)).days
                if file_age_days > 7:
                    cache_file.unlink()
                    logger.debug(f"Removed old cache: {cache_file}")
        except Exception as e:
            logger.debug(f"Cache cleanup error: {e}")
            
    def _safe_query(self, client: bigquery.Client, query: str, 
                   job_config: Optional[bigquery.QueryJobConfig] = None,
                   location: Optional[str] = None, timeout: int = 30) -> Optional[bigquery.QueryJob]:
        """Execute query with better error handling"""
        
        query_hash = hashlib.md5(query.encode()).hexdigest()[:8]
        
        if query_hash in self.failed_queries:
            logger.debug(f"Skipping previously failed query {query_hash}")
            return None
            
        if job_config is None:
            job_config = bigquery.QueryJobConfig()
            job_config.use_query_cache = True
            
        try:
            # Try with location if provided
            if location:
                query_job = client.query(query, job_config=job_config, location=location)
            else:
                query_job = client.query(query, job_config=job_config)
                
            results = query_job.result(timeout=timeout)
            return results
            
        except exceptions.BadRequest as e:
            error_msg = str(e)
            
            # Handle specific errors
            if 'Cannot query over table' in error_msg and 'without a filter' in error_msg:
                logger.warning(f"Table requires partition filter: {error_msg[:100]}")
                # Try to add a date filter if it's a partitioned table
                if '_TABLE_SUFFIX' not in query and 'WHERE' not in query.upper():
                    # Add a recent date filter
                    modified_query = query.replace('FROM', 'FROM') + ' WHERE DATE(_PARTITIONTIME) >= DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY)'
                    return self._safe_query(client, modified_query, job_config, location, timeout)
                    
            elif 'Syntax error' in error_msg:
                logger.warning(f"Query syntax error: {error_msg[:200]}")
                
            elif 'does not have bigquery.tables.getData permission' in error_msg:
                logger.warning(f"Permission denied for table")
                
            else:
                logger.warning(f"Query failed: {error_msg[:200]}")
                
            self.failed_queries.add(query_hash)
            return None
            
        except exceptions.Forbidden as e:
            logger.warning(f"Access forbidden: {str(e)[:100]}")
            self.failed_queries.add(query_hash)
            return None
            
        except exceptions.NotFound as e:
            logger.warning(f"Resource not found: {str(e)[:100]}")
            self.failed_queries.add(query_hash)
            return None
            
        except Exception as e:
            logger.warning(f"Query error: {str(e)[:100]}")
            self.failed_queries.add(query_hash)
            return None
            
    async def execute(self):
        logger.info(f"Starting CMDB+ discovery for projects: {self.project_ids}")
        
        # If we have fresh cache and user accepted it, skip to building CMDB
        if self.using_cache and len(self.discovered_hosts) > 0:
            logger.info("Using cached data, skipping discovery phases")
            await self._build_cmdb()
            self._report_statistics()
            return
        
        try:
            # Phase 1: Learn patterns
            await self._learn_hostname_patterns()
            
            # Phase 2: Discover all data
            await self._discover_all_data()
            
            # Phase 3: Analyze
            await self._analyze_and_aggregate()
            
            # Save daily cache for future runs
            self._save_daily_cache()
            
            # Phase 4: Build CMDB
            await self._build_cmdb()
            
            # Report
            self._report_statistics()
            
            # Clean up checkpoint on success
            if self.checkpoint_file.exists():
                self.checkpoint_file.unlink()
                logger.info("Checkpoint file removed after successful completion")
                
        except Exception as e:
            logger.error(f"CMDB+ execution failed: {e}", exc_info=True)
            self._save_checkpoint()
            raise
        finally:
            self.executor.shutdown(wait=True)
            
    async def _learn_hostname_patterns(self):
        """Phase 1: Quick pattern learning"""
        logger.info("Phase 1: Learning hostname patterns...")
        
        training_samples = []
        max_samples = self.config.get('max_training_samples', 50000)
        max_time = 180  # 3 minutes max for training
        start_time = time.time()
        
        for project_id in self.project_ids:
            if len(training_samples) >= max_samples or (time.time() - start_time) > max_time:
                break
                
            try:
                client = self.bq_manager.get_client(project_id)
                datasets = list(client.list_datasets())
                
                for dataset_ref in datasets[:5]:  # Sample first 5 datasets
                    dataset_id = dataset_ref.dataset_id
                    
                    try:
                        tables = list(client.list_tables(dataset_ref))
                        
                        for table_ref in tables[:10]:  # Sample first 10 tables
                            table_id = table_ref.table_id
                            
                            try:
                                table = client.get_table(table_ref)
                                
                                # Find potential hostname columns
                                for field in table.schema:
                                    field_lower = field.name.lower()
                                    if any(term in field_lower for term in ['host', 'server', 'machine', 'node']):
                                        
                                        # Get samples
                                        query = f"""
                                        SELECT DISTINCT `{field.name}` as value
                                        FROM `{project_id}.{dataset_id}.{table_id}`
                                        WHERE `{field.name}` IS NOT NULL
                                        LIMIT 500
                                        """
                                        
                                        results = self._safe_query(client, query, timeout=10)
                                        if results:
                                            for row in results:
                                                if row.value and len(training_samples) < max_samples:
                                                    training_samples.append(row.value)
                                                    
                            except Exception as e:
                                logger.debug(f"Skipping table {table_id}: {e}")
                                
                    except Exception as e:
                        logger.debug(f"Skipping dataset {dataset_id}: {e}")
                        
            except Exception as e:
                logger.error(f"Error in project {project_id}: {e}")
                
        if training_samples:
            logger.info(f"Training on {len(training_samples)} samples")
            await self.discovery_engine.train(training_samples)
            await self.pattern_model.learn_patterns(training_samples)
        else:
            logger.warning("No training samples found, using heuristics")
            
    async def _discover_all_data(self):
        """Phase 2: Discover ALL data from ALL tables"""
        logger.info("Phase 2: Discovering all hosts and data...")
        
        for project_id in self.project_ids:
            if self.shutdown_requested:
                break
                
            try:
                client = self.bq_manager.get_client(project_id)
                datasets = list(client.list_datasets())
                self.scan_metrics['projects'] += 1
                
                logger.info(f"Processing {len(datasets)} datasets in {project_id}")
                
                for dataset_ref in datasets:
                    if self.shutdown_requested:
                        break
                        
                    dataset_id = dataset_ref.dataset_id
                    self.scan_metrics['datasets'] += 1
                    
                    try:
                        tables = list(client.list_tables(dataset_ref))
                        logger.info(f"Processing {len(tables)} tables in {dataset_id}")
                        
                        for table_ref in tables:
                            if self.shutdown_requested:
                                break
                                
                            table_id = table_ref.table_id
                            full_table_name = f"{project_id}.{dataset_id}.{table_id}"
                            
                            # Skip if already processed
                            if full_table_name in self.processed_tables:
                                continue
                                
                            self._process_table(client, project_id, dataset_id, table_id, full_table_name)
                            
                    except Exception as e:
                        logger.error(f"Error processing dataset {dataset_id}: {e}")
                        self.scan_metrics['errors'] += 1
                        
            except Exception as e:
                logger.error(f"Error processing project {project_id}: {e}")
                self.scan_metrics['errors'] += 1
                
    def _process_table(self, client: bigquery.Client, project_id: str, 
                      dataset_id: str, table_id: str, full_table_name: str):
        """Process a table - ALL rows, not just samples"""
        
        try:
            self.scan_metrics['tables'] += 1
            table = client.get_table(f"{project_id}.{dataset_id}.{table_id}")
            total_rows = table.num_rows or 0
            
            if total_rows == 0:
                logger.debug(f"Table {table_id} is empty")
                self.processed_tables.add(full_table_name)
                return
                
            logger.info(f"Processing table {self.scan_metrics['tables']}: {full_table_name} ({total_rows:,} rows)")
            
            # Process ALL rows in batches
            processed_rows = 0
            batch_size = self.config.get('rows_per_batch', 10000)
            
            while processed_rows < total_rows:
                if self.shutdown_requested:
                    break
                    
                # Use LIMIT/OFFSET to get ALL data
                query = f"""
                SELECT *
                FROM `{project_id}.{dataset_id}.{table_id}`
                LIMIT {batch_size}
                OFFSET {processed_rows}
                """
                
                results = self._safe_query(client, query, timeout=60)
                
                if results is None:
                    logger.warning(f"Failed to query {table_id}, trying alternative approach")
                    
                    # Try without OFFSET (some tables don't support it)
                    if processed_rows == 0:
                        query = f"""
                        SELECT *
                        FROM `{project_id}.{dataset_id}.{table_id}`
                        LIMIT {min(batch_size, total_rows)}
                        """
                        results = self._safe_query(client, query, timeout=60)
                        
                    if results is None:
                        logger.error(f"Skipping table {table_id} after all attempts failed")
                        self.scan_metrics['errors'] += 1
                        break
                        
                batch_count = 0
                for row in results:
                    self.scan_metrics['rows'] += 1
                    batch_count += 1
                    
                    # Process each row
                    self._process_row(dict(row), full_table_name)
                    
                    # Save checkpoint periodically
                    if self.scan_metrics['rows'] % self.checkpoint_interval == 0:
                        self._save_checkpoint()
                        gc.collect()
                        
                processed_rows += batch_count
                
                if batch_count == 0:
                    break
                    
                if batch_count < batch_size:
                    # Got less than requested, probably at end of table
                    break
                    
                logger.debug(f"Processed {processed_rows:,}/{total_rows:,} rows from {table_id}")
                
            self.processed_tables.add(full_table_name)
            logger.info(f"Completed {full_table_name}: processed {processed_rows:,} rows")
            
        except Exception as e:
            logger.error(f"Error processing table {full_table_name}: {e}")
            self.scan_metrics['errors'] += 1
            self.processed_tables.add(full_table_name)  # Mark as processed to avoid retry
            
    def _process_row(self, row: Dict[str, Any], table_name: str):
        """Process a single row"""
        potential_hosts = []
        row_normalized = {}
        
        for column, value in row.items():
            if value is None:
                continue
                
            # Track column importance
            self.column_importance[column] += 1
            
            # Observe for classification
            self.column_classifier.observe(column, value, table_name)
            
            # Normalize value
            normalized = self.normalizer.normalize(value)
            raw_value = str(value)
            
            row_normalized[column] = {'raw': raw_value, 'normalized': normalized}
            
            # Check if this could be a hostname
            if self.discovery_engine.is_hostname(normalized):
                confidence = self.discovery_engine.get_confidence(normalized)
                potential_hosts.append({
                    'raw': raw_value,
                    'normalized': self.normalizer.normalize_hostname(normalized),
                    'column': column,
                    'confidence': confidence
                })
                
        # Store discovered hosts with associated data
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
            
            # Track occurrences
            max_occurrences = self.config.get('max_occurrences_per_host', 100)
            if len(self.discovered_hosts[normalized_host]['occurrences']) < max_occurrences:
                self.discovered_hosts[normalized_host]['occurrences'].append({
                    'table': table_name,
                    'column': host_info['column'],
                    'confidence': host_info['confidence']
                })
                
            # Store associated data from same row
            max_values = self.config.get('max_values_per_column', 50)
            for col, values in row_normalized.items():
                if col != host_info['column']:
                    if len(self.discovered_hosts[normalized_host]['associated_data'][col]['raw']) < max_values:
                        self.discovered_hosts[normalized_host]['associated_data'][col]['raw'].add(values['raw'])
                    if len(self.discovered_hosts[normalized_host]['associated_data'][col]['normalized']) < max_values:
                        self.discovered_hosts[normalized_host]['associated_data'][col]['normalized'].add(values['normalized'])
                        
    async def _analyze_and_aggregate(self):
        """Phase 3: Analyze collected data"""
        logger.info("Phase 3: Analyzing and aggregating...")
        
        # Discover column types
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
        
        # Limit columns
        column_limit = self.config.get('column_limit', 100)
        self.important_columns = self.important_columns[:column_limit]
        
        logger.info(f"Identified {len(self.important_columns)} important columns")
        logger.info(f"Top 20 columns: {[c['name'] for c in self.important_columns[:20]]}")
        
    async def _build_cmdb(self):
        """Phase 4: Build the CMDB database"""
        logger.info("Phase 4: Building CMDB...")
        
        await self.cmdb_builder.initialize()
        
        # Define schema
        schema_columns = ['hostname', 'raw_forms', 'occurrence_count', 'confidence']
        
        # Add important columns
        column_limit = self.config.get('column_limit', 100)
        for col_info in self.important_columns[:column_limit]:
            schema_columns.append(col_info['name'])
            
        await self.cmdb_builder.create_hosts_table(schema_columns)
        
        # Process hosts in batches
        host_records = []
        batch_size = 1000
        
        for i, (normalized_host, data) in enumerate(self.discovered_hosts.items()):
            record = {
                'hostname': normalized_host,
                'raw_forms': json.dumps(list(data['raw_forms'])[:20]),
                'occurrence_count': len(data['occurrences']),
                'confidence': max([o['confidence'] for o in data['occurrences']] or [0])
            }
            
            # Add associated data
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
            
            # Insert in batches
            if len(host_records) >= batch_size:
                await self.cmdb_builder.bulk_insert('hosts', host_records)
                host_records = []
                gc.collect()
                
        # Insert remaining
        if host_records:
            await self.cmdb_builder.bulk_insert('hosts', host_records)
            
        # Create indexes
        index_columns = ['hostname', 'occurrence_count', 'confidence']
        index_columns.extend([c['name'] for c in self.important_columns[:20]])
        
        await self.cmdb_builder.create_indexes(index_columns)
        
        # Export backup
        self.cmdb_builder.export_to_csv('cmdb_backup.csv')
        
        logger.info(f"CMDB built with {len(self.discovered_hosts)} unique hosts")
        
    def _report_statistics(self):
        """Generate final report"""
        elapsed = (datetime.now() - self.scan_metrics['start_time']).total_seconds()
        
        stats_report = f"""
{"="*60}
CMDB+ DISCOVERY COMPLETE
{"="*60}
Projects scanned: {self.scan_metrics['projects']}
Datasets scanned: {self.scan_metrics['datasets']}
Tables scanned: {self.scan_metrics['tables']}
Rows processed: {self.scan_metrics['rows']:,}
Unique hosts discovered: {len(self.discovered_hosts):,}
Important columns identified: {len(self.important_columns)}
Errors encountered: {self.scan_metrics.get('errors', 0)}
Failed queries: {len(self.failed_queries)}
Time elapsed: {elapsed:.1f} seconds
Processing rate: {self.scan_metrics['rows']/elapsed:.0f} rows/second
Cache status: {'Used cached data' if self.using_cache else 'Fresh scan'}
{"="*60}
"""
        logger.info(stats_report)
        
        # Save statistics
        stats_file = Path('cmdb_statistics.json')
        with open(stats_file, 'w') as f:
            json.dump({
                'scan_metrics': self.scan_metrics,
                'hosts_discovered': len(self.discovered_hosts),
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
            'max_memory_mb': 8192,
            'rows_per_batch': 10000,
            'full_scan_large_tables': True,
            'use_daily_cache': True,
            'cache_expiry_hours': 24
        }
        
        with open(config_file, 'w') as f:
            json.dump(default_config, f, indent=2)
            
        logger.error(f"Please edit {config_file} with your project IDs")
        sys.exit(1)
        
    with open(config_file, 'r') as f:
        config = json.load(f)
        
    projects = config.get('projects', [])
    
    if not projects or projects == ['your-project-id-1', 'your-project-id-2']:
        logger.error(f"Please edit {config_file} with your actual BigQuery project IDs")
        sys.exit(1)
        
    orchestrator = CMDBPlusOrchestrator(projects, config)
    await orchestrator.execute()

if __name__ == "__main__":
    asyncio.run(main())