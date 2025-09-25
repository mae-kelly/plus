import os
import sys
import json
import asyncio
import pickle
import signal
from pathlib import Path
from typing import Dict, List, Set, Any
from collections import defaultdict
from datetime import datetime
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc

from bigquery_client import BigQueryClientManager
from discovery_engine import DiscoveryEngine
from normalization_engine import NormalizationEngine
from pattern_recognition import PatternRecognitionModel
from column_classifier import ColumnClassifier
from data_aggregator import DataAggregator
from cmdb_builder import CMDBBuilder
from gpu_accelerator import GPUAccelerator
from relationship_analyzer import RelationshipAnalyzer

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
        
        # Checkpoint file for resuming
        self.checkpoint_file = Path('cmdb_checkpoint.pkl')
        self.checkpoint_interval = 1000  # Save every 1000 rows
        
        # Force GPU
        self.gpu = GPUAccelerator()
        self.device = self.gpu.initialize()
        if 'cpu' in str(self.device).lower():
            raise RuntimeError("GPU required. No CPU fallback.")
            
        self._initialize_components()
        
        # Load checkpoint if exists
        self._load_checkpoint()
        
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
                'retry_delay_seconds': 5
            }
            # Save default config
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
        
        self.bq_manager = BigQueryClientManager()
        self.discovery_engine = DiscoveryEngine(device=self.device)
        self.normalizer = NormalizationEngine()
        self.pattern_model = PatternRecognitionModel(device=self.device)
        self.column_classifier = ColumnClassifier(device=self.device)
        self.aggregator = DataAggregator()
        self.relationship_analyzer = RelationshipAnalyzer()
        self.cmdb_builder = CMDBBuilder('new_cmdb.db')
        
        logger.info("All components initialized")
        
    def _load_checkpoint(self):
        if self.checkpoint_file.exists() and self.config.get('checkpoint_enabled', True):
            try:
                with open(self.checkpoint_file, 'rb') as f:
                    checkpoint = pickle.load(f)
                    
                self.discovered_hosts = checkpoint.get('discovered_hosts', {})
                self.column_importance = checkpoint.get('column_importance', defaultdict(int))
                self.scan_metrics = checkpoint.get('scan_metrics', {
                    'projects': 0, 'datasets': 0, 'tables': 0, 'rows': 0, 
                    'hosts': 0, 'start_time': datetime.now()
                })
                self.processed_tables = checkpoint.get('processed_tables', set())
                
                logger.info(f"Checkpoint loaded: {len(self.discovered_hosts)} hosts, {len(self.processed_tables)} tables processed")
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}")
                self._init_tracking()
        else:
            self._init_tracking()
            
    def _init_tracking(self):
        self.discovered_hosts = {}
        self.column_importance = defaultdict(int)
        self.scan_metrics = {
            'projects': 0, 'datasets': 0, 'tables': 0, 'rows': 0, 
            'hosts': 0, 'start_time': datetime.now(), 'errors': 0
        }
        self.processed_tables = set()
        
    def _save_checkpoint(self):
        if not self.config.get('checkpoint_enabled', True):
            return
            
        checkpoint = {
            'discovered_hosts': self.discovered_hosts,
            'column_importance': dict(self.column_importance),
            'scan_metrics': self.scan_metrics,
            'processed_tables': self.processed_tables
        }
        
        temp_file = self.checkpoint_file.with_suffix('.tmp')
        try:
            with open(temp_file, 'wb') as f:
                pickle.dump(checkpoint, f, protocol=pickle.HIGHEST_PROTOCOL)
            temp_file.replace(self.checkpoint_file)
            logger.debug(f"Checkpoint saved: {len(self.discovered_hosts)} hosts")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            
    async def execute(self):
        logger.info(f"Starting CMDB+ discovery for projects: {self.project_ids}")
        
        try:
            await self._learn_hostname_patterns()
            await self._discover_all_data()
            await self._analyze_and_aggregate()
            await self._build_cmdb()
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
        logger.info("Phase 1: Learning hostname patterns from host/hostname columns...")
        
        training_samples = []
        
        for project_id in self.project_ids:
            if self.shutdown_requested:
                break
                
            self.scan_metrics['projects'] += 1
            
            for attempt in range(self.config.get('retry_attempts', 3)):
                try:
                    client = self.bq_manager.get_client(project_id)
                    datasets = list(client.list_datasets())
                    
                    for dataset_ref in datasets:
                        if self.shutdown_requested:
                            break
                            
                        self.scan_metrics['datasets'] += 1
                        dataset_id = dataset_ref.dataset_id
                        
                        try:
                            tables = list(client.list_tables(dataset_ref))
                            
                            for table_ref in tables:
                                self.scan_metrics['tables'] += 1
                                table_id = table_ref.table_id
                                full_table_name = f"{project_id}.{dataset_id}.{table_id}"
                                
                                # Skip if already processed
                                if full_table_name in self.processed_tables:
                                    continue
                                    
                                try:
                                    table = client.get_table(table_ref)
                                    
                                    # Look ONLY for host/hostname columns
                                    for field in table.schema:
                                        if field.name.lower() in ['host', 'hostname']:
                                            logger.info(f"Found training column: {field.name} in {full_table_name}")
                                            
                                            # Use TABLESAMPLE for large tables
                                            sample_query = f"""
                                            SELECT DISTINCT {field.name}
                                            FROM `{project_id}.{dataset_id}.{table_id}`
                                            TABLESAMPLE SYSTEM (10 PERCENT)
                                            WHERE {field.name} IS NOT NULL
                                            LIMIT 10000
                                            """
                                            
                                            try:
                                                results = client.query(sample_query).result()
                                                for row in results:
                                                    value = row[field.name]
                                                    if value:
                                                        training_samples.append(value)
                                            except Exception as e:
                                                logger.warning(f"Failed to query {table_id}: {e}")
                                                self.scan_metrics['errors'] += 1
                                                
                                except Exception as e:
                                    logger.warning(f"Failed to process table {table_id}: {e}")
                                    self.scan_metrics['errors'] += 1
                                    
                        except Exception as e:
                            logger.warning(f"Failed to list tables in dataset {dataset_id}: {e}")
                            self.scan_metrics['errors'] += 1
                            
                    break  # Success, exit retry loop
                    
                except Exception as e:
                    logger.error(f"Attempt {attempt+1} failed for project {project_id}: {e}")
                    if attempt < self.config.get('retry_attempts', 3) - 1:
                        await asyncio.sleep(self.config.get('retry_delay_seconds', 5))
                    else:
                        logger.error(f"Failed to process project {project_id} after all retries")
                        self.scan_metrics['errors'] += 1
                        
        if training_samples:
            logger.info(f"Training on {len(training_samples)} hostname samples")
            await self.discovery_engine.train(training_samples)
            await self.pattern_model.learn_patterns(training_samples)
        else:
            logger.warning("No training samples found, using heuristics")
            
    async def _discover_all_data(self):
        logger.info("Phase 2: Discovering all hosts and data...")
        
        for project_id in self.project_ids:
            if self.shutdown_requested:
                break
                
            for attempt in range(self.config.get('retry_attempts', 3)):
                try:
                    await self._process_project(project_id)
                    break
                except Exception as e:
                    logger.error(f"Attempt {attempt+1} failed for project {project_id}: {e}")
                    if attempt < self.config.get('retry_attempts', 3) - 1:
                        await asyncio.sleep(self.config.get('retry_delay_seconds', 5))
                    else:
                        self.scan_metrics['errors'] += 1
                        
    async def _process_project(self, project_id: str):
        client = self.bq_manager.get_client(project_id)
        datasets = list(client.list_datasets())
        
        for dataset_ref in datasets:
            if self.shutdown_requested:
                break
                
            dataset_id = dataset_ref.dataset_id
            tables = list(client.list_tables(dataset_ref))
            
            # Process tables in parallel batches
            table_futures = []
            
            for table_ref in tables:
                if self.shutdown_requested:
                    break
                    
                table_id = table_ref.table_id
                full_table_name = f"{project_id}.{dataset_id}.{table_id}"
                
                # Skip if already processed
                if full_table_name in self.processed_tables:
                    logger.debug(f"Skipping already processed table: {full_table_name}")
                    continue
                    
                future = self.executor.submit(
                    self._process_table_sync,
                    client, project_id, dataset_id, table_id, full_table_name
                )
                table_futures.append(future)
                
                # Process in batches to avoid overwhelming memory
                if len(table_futures) >= self.config.get('max_workers', 5):
                    for future in as_completed(table_futures):
                        try:
                            future.result()
                        except Exception as e:
                            logger.error(f"Table processing failed: {e}")
                            self.scan_metrics['errors'] += 1
                    table_futures = []
                    
            # Process remaining futures
            for future in as_completed(table_futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Table processing failed: {e}")
                    self.scan_metrics['errors'] += 1
                    
    def _process_table_sync(self, client, project_id: str, dataset_id: str, 
                           table_id: str, full_table_name: str):
        try:
            logger.info(f"Processing table: {full_table_name}")
            
            table = client.get_table(f"{project_id}.{dataset_id}.{table_id}")
            total_rows = table.num_rows
            
            if total_rows == 0:
                self.processed_tables.add(full_table_name)
                return
                
            # For very large tables, use sampling
            if total_rows > 1000000:
                logger.info(f"Large table detected ({total_rows:,} rows), using sampling")
                query = f"""
                SELECT *
                FROM `{project_id}.{dataset_id}.{table_id}`
                TABLESAMPLE SYSTEM (10 PERCENT)
                """
            else:
                # Process entire table in batches
                processed_rows = 0
                
                while processed_rows < total_rows:
                    if self.shutdown_requested:
                        break
                        
                    # Use ROW_NUMBER for efficient pagination
                    query = f"""
                    WITH numbered_rows AS (
                        SELECT *, ROW_NUMBER() OVER() as rn
                        FROM `{project_id}.{dataset_id}.{table_id}`
                    )
                    SELECT * EXCEPT(rn)
                    FROM numbered_rows
                    WHERE rn > {processed_rows} AND rn <= {processed_rows + self.rows_per_batch}
                    """
                    
                    try:
                        results = client.query(query).result()
                        batch_count = 0
                        
                        for row in results:
                            self.scan_metrics['rows'] += 1
                            batch_count += 1
                            
                            # Process row synchronously
                            self._process_row_sync(dict(row), full_table_name, table.schema)
                            
                            # Save checkpoint periodically
                            if self.scan_metrics['rows'] % self.checkpoint_interval == 0:
                                self._save_checkpoint()
                                # Memory cleanup
                                gc.collect()
                                
                        processed_rows += batch_count
                        
                        if batch_count == 0:
                            break  # No more rows
                            
                        logger.debug(f"Processed {processed_rows:,}/{total_rows:,} rows from {table_id}")
                        
                    except Exception as e:
                        logger.warning(f"Failed to query batch from {table_id}: {e}")
                        self.scan_metrics['errors'] += 1
                        break
                        
            self.processed_tables.add(full_table_name)
            
        except Exception as e:
            logger.error(f"Failed to process table {full_table_name}: {e}")
            self.scan_metrics['errors'] += 1
            
    def _process_row_sync(self, row: Dict[str, Any], table_name: str, schema):
        potential_hosts = []
        row_normalized = {}
        
        for column, value in row.items():
            if value is None:
                continue
                
            # Observe column for classification
            self.column_classifier.observe(column, value, table_name)
            
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
                
            self.column_importance[column] += 1
            
        # Store discovered hosts with all associated data
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
            
            # Limit occurrences to prevent memory issues
            if len(self.discovered_hosts[normalized_host]['occurrences']) < 100:
                self.discovered_hosts[normalized_host]['occurrences'].append({
                    'table': table_name,
                    'column': host_info['column'],
                    'confidence': host_info['confidence']
                })
            
            # Store ALL other column data from this row
            for col, values in row_normalized.items():
                if col != host_info['column']:
                    # Limit stored values to prevent memory issues
                    if len(self.discovered_hosts[normalized_host]['associated_data'][col]['raw']) < 50:
                        self.discovered_hosts[normalized_host]['associated_data'][col]['raw'].add(values['raw'])
                    if len(self.discovered_hosts[normalized_host]['associated_data'][col]['normalized']) < 50:
                        self.discovered_hosts[normalized_host]['associated_data'][col]['normalized'].add(values['normalized'])
                        
    async def _analyze_and_aggregate(self):
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
        
        logger.info(f"Identified {len(self.important_columns)} important columns")
        logger.info(f"Top 20 columns: {[c['name'] for c in self.important_columns[:20]]}")
        
    async def _build_cmdb(self):
        logger.info("Phase 4: Building CMDB...")
        
        await self.cmdb_builder.initialize()
        
        schema_columns = ['hostname', 'raw_forms', 'occurrence_count', 'confidence']
        
        # Add top 100 most important columns dynamically
        for col_info in self.important_columns[:100]:
            schema_columns.append(col_info['name'])
            
        await self.cmdb_builder.create_hosts_table(schema_columns)
        
        # Process hosts in batches to manage memory
        host_records = []
        batch_size = 1000
        
        for i, (normalized_host, data) in enumerate(self.discovered_hosts.items()):
            record = {
                'hostname': normalized_host,
                'raw_forms': json.dumps(list(data['raw_forms'])[:20]),  # Limit raw forms
                'occurrence_count': len(data['occurrences']),
                'confidence': max([o['confidence'] for o in data['occurrences']] or [0])
            }
            
            # Add all discovered associated data
            for col_info in self.important_columns[:100]:
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
                
        # Insert remaining records
        if host_records:
            await self.cmdb_builder.bulk_insert('hosts', host_records)
            
        # Create indexes
        index_columns = ['hostname', 'occurrence_count', 'confidence']
        index_columns.extend([c['name'] for c in self.important_columns[:20]])
        
        await self.cmdb_builder.create_indexes(index_columns)
        
        # Export to CSV for backup
        self.cmdb_builder.export_to_csv('cmdb_backup.csv')
        
        logger.info(f"CMDB built with {len(self.discovered_hosts)} unique hosts")
        
    def _report_statistics(self):
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
Time elapsed: {elapsed:.1f} seconds
Processing rate: {self.scan_metrics['rows']/elapsed:.0f} rows/second
{"="*60}
"""
        logger.info(stats_report)
        
        # Save statistics to file
        stats_file = Path('cmdb_statistics.json')
        with open(stats_file, 'w') as f:
            json.dump({
                'scan_metrics': self.scan_metrics,
                'hosts_discovered': len(self.discovered_hosts),
                'important_columns': [c['name'] for c in self.important_columns[:50]],
                'elapsed_seconds': elapsed
            }, f, indent=2, default=str)

async def main():
    # Load configuration
    config_file = Path('config.json')
    
    if not config_file.exists():
        # Create default config
        default_config = {
            'projects': ['your-project-id-1', 'your-project-id-2'],
            'max_workers': 5,
            'max_memory_mb': 8192,
            'rows_per_batch': 10000,
            'checkpoint_enabled': True,
            'retry_attempts': 3,
            'retry_delay_seconds': 5
        }
        
        with open(config_file, 'w') as f:
            json.dump(default_config, f, indent=2)
            
        logger.error(f"Please edit {config_file} with your project IDs and run again")
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