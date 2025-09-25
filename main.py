import os
import sys
import json
import asyncio
from pathlib import Path
from typing import Dict, List, Set, Any
from collections import defaultdict
from datetime import datetime
import logging

from bigquery_client import BigQueryClientManager
from discovery_engine import DiscoveryEngine
from normalization_engine import NormalizationEngine
from pattern_recognition import PatternRecognitionModel
from column_classifier import ColumnClassifier
from data_aggregator import DataAggregator
from cmdb_builder import CMDBBuilder
from gpu_accelerator import GPUAccelerator
from relationship_analyzer import RelationshipAnalyzer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CMDBPlusOrchestrator:
    def __init__(self, project_ids: List[str]):
        self.project_ids = project_ids
        
        # Force GPU
        self.gpu = GPUAccelerator()
        self.device = self.gpu.initialize()
        if 'cpu' in str(self.device).lower():
            raise RuntimeError("GPU required. No CPU fallback.")
            
        self._initialize_components()
        
        # Tracking
        self.discovered_hosts = {}
        self.column_importance = defaultdict(int)
        self.scan_metrics = {'projects': 0, 'datasets': 0, 'tables': 0, 'rows': 0, 'hosts': 0, 'start_time': datetime.now()}
        
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
        
    async def execute(self):
        logger.info(f"Starting CMDB+ discovery for projects: {self.project_ids}")
        
        try:
            await self._learn_hostname_patterns()
            await self._discover_all_data()
            await self._analyze_and_aggregate()
            await self._build_cmdb()
            self._report_statistics()
        except Exception as e:
            logger.error(f"CMDB+ execution failed: {e}")
            raise
            
    async def _learn_hostname_patterns(self):
        logger.info("Phase 1: Learning hostname patterns from host/hostname columns...")
        
        training_samples = []
        
        for project_id in self.project_ids:
            self.scan_metrics['projects'] += 1
            client = self.bq_manager.get_client(project_id)
            
            try:
                datasets = list(client.list_datasets())
                
                for dataset_ref in datasets:
                    self.scan_metrics['datasets'] += 1
                    dataset_id = dataset_ref.dataset_id
                    tables = list(client.list_tables(dataset_ref))
                    
                    for table_ref in tables:
                        self.scan_metrics['tables'] += 1
                        table_id = table_ref.table_id
                        table = client.get_table(table_ref)
                        
                        # Look ONLY for host/hostname columns
                        for field in table.schema:
                            if field.name.lower() in ['host', 'hostname']:
                                logger.info(f"Found training column: {field.name} in {project_id}.{dataset_id}.{table_id}")
                                
                                query = f"""
                                SELECT DISTINCT {field.name}
                                FROM `{project_id}.{dataset_id}.{table_id}`
                                WHERE {field.name} IS NOT NULL
                                LIMIT 10000
                                """
                                
                                try:
                                    results = client.query(query).result()
                                    for row in results:
                                        value = row[field.name]
                                        if value:
                                            training_samples.append(value)
                                except Exception as e:
                                    logger.warning(f"Failed to query {table_id}: {e}")
                                    
            except Exception as e:
                logger.error(f"Failed to process project {project_id}: {e}")
                
        if training_samples:
            logger.info(f"Training on {len(training_samples)} hostname samples")
            await self.discovery_engine.train(training_samples)
            await self.pattern_model.learn_patterns(training_samples)
        else:
            logger.warning("No training samples found, using heuristics")
            
    async def _discover_all_data(self):
        logger.info("Phase 2: Discovering all hosts and data...")
        
        for project_id in self.project_ids:
            client = self.bq_manager.get_client(project_id)
            
            try:
                datasets = list(client.list_datasets())
                
                for dataset_ref in datasets:
                    dataset_id = dataset_ref.dataset_id
                    tables = list(client.list_tables(dataset_ref))
                    
                    for table_ref in tables:
                        table_id = table_ref.table_id
                        full_table_name = f"{project_id}.{dataset_id}.{table_id}"
                        
                        logger.info(f"Scanning: {full_table_name}")
                        
                        table = client.get_table(table_ref)
                        total_rows = table.num_rows
                        
                        if total_rows == 0:
                            continue
                            
                        # Process ALL rows in chunks
                        chunk_size = 100000
                        offset = 0
                        
                        while offset < total_rows:
                            query = f"""
                            SELECT *
                            FROM `{project_id}.{dataset_id}.{table_id}`
                            LIMIT {chunk_size}
                            OFFSET {offset}
                            """
                            
                            try:
                                results = client.query(query).result()
                                
                                for row in results:
                                    self.scan_metrics['rows'] += 1
                                    await self._process_row(dict(row), full_table_name, table.schema)
                                    
                            except Exception as e:
                                logger.warning(f"Failed to query chunk from {table_id}: {e}")
                                
                            offset += chunk_size
                            
                            if self.scan_metrics['rows'] % 1000000 == 0:
                                logger.info(f"Progress: {self.scan_metrics['rows']:,} rows, {len(self.discovered_hosts):,} hosts")
                                          
            except Exception as e:
                logger.error(f"Failed to scan project {project_id}: {e}")
                
    async def _process_row(self, row: Dict[str, Any], table_name: str, schema):
        potential_hosts = []
        row_normalized = {}
        
        for column, value in row.items():
            if value is None:
                continue
                
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
            
            if not normalized_host:
                continue
                
            if normalized_host not in self.discovered_hosts:
                self.discovered_hosts[normalized_host] = {
                    'raw_forms': set(),
                    'occurrences': [],
                    'associated_data': defaultdict(lambda: defaultdict(set))
                }
                
            self.discovered_hosts[normalized_host]['raw_forms'].add(host_info['raw'])
            
            self.discovered_hosts[normalized_host]['occurrences'].append({
                'table': table_name,
                'column': host_info['column'],
                'confidence': host_info['confidence']
            })
            
            # Store ALL other column data from this row
            for col, values in row_normalized.items():
                if col != host_info['column']:
                    self.discovered_hosts[normalized_host]['associated_data'][col]['raw'].add(values['raw'])
                    self.discovered_hosts[normalized_host]['associated_data'][col]['normalized'].add(values['normalized'])
                    
    async def _analyze_and_aggregate(self):
        logger.info("Phase 3: Analyzing and aggregating...")
        
        total_occurrences = sum(self.column_importance.values())
        
        self.important_columns = []
        for column, count in self.column_importance.items():
            importance = count / total_occurrences
            
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
        
        host_records = []
        
        for normalized_host, data in self.discovered_hosts.items():
            record = {
                'hostname': normalized_host,
                'raw_forms': json.dumps(list(data['raw_forms'])),
                'occurrence_count': len(data['occurrences']),
                'confidence': max([o['confidence'] for o in data['occurrences']] or [0])
            }
            
            # Add all discovered associated data
            for col_info in self.important_columns[:100]:
                col_name = col_info['name']
                
                if col_name in data['associated_data']:
                    normalized_values = list(data['associated_data'][col_name]['normalized'])
                    
                    if len(normalized_values) == 1:
                        record[col_name] = normalized_values[0]
                    elif len(normalized_values) > 1:
                        record[col_name] = json.dumps(normalized_values[:10])
                    else:
                        record[col_name] = None
                else:
                    record[col_name] = None
                    
            host_records.append(record)
            
        await self.cmdb_builder.bulk_insert('hosts', host_records)
        
        index_columns = ['hostname', 'occurrence_count', 'confidence']
        index_columns.extend([c['name'] for c in self.important_columns[:20]])
        
        await self.cmdb_builder.create_indexes(index_columns)
        
        logger.info(f"CMDB built with {len(host_records)} unique hosts")
        
    def _report_statistics(self):
        elapsed = (datetime.now() - self.scan_metrics['start_time']).total_seconds()
        
        logger.info("="*60)
        logger.info("CMDB+ DISCOVERY COMPLETE")
        logger.info("="*60)
        logger.info(f"Projects scanned: {self.scan_metrics['projects']}")
        logger.info(f"Datasets scanned: {self.scan_metrics['datasets']}")
        logger.info(f"Tables scanned: {self.scan_metrics['tables']}")
        logger.info(f"Rows processed: {self.scan_metrics['rows']:,}")
        logger.info(f"Unique hosts discovered: {len(self.discovered_hosts):,}")
        logger.info(f"Important columns identified: {len(self.important_columns)}")
        logger.info(f"Time elapsed: {elapsed:.1f} seconds")
        logger.info(f"Processing rate: {self.scan_metrics['rows']/elapsed:.0f} rows/second")
        logger.info("="*60)

async def main():
    # Specify your project IDs here
    projects = ['your-project-id-1', 'your-project-id-2']
    
    orchestrator = CMDBPlusOrchestrator(projects)
    await orchestrator.execute()

if __name__ == "__main__":
    asyncio.run(main())