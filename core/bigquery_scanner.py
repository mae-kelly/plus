# core/bigquery_scanner.py
"""
ULTRA-FAST BigQuery Scanner - Uses INFORMATION_SCHEMA instead of API calls
"""

import asyncio
from typing import Dict, List, Any, Optional, Set
import logging
from datetime import datetime
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

from google.cloud import bigquery
from core.bigquery_client_manager import BigQueryClientManager

logger = logging.getLogger(__name__)

class BigQueryScanner:
    """ULTRA-FAST scanner using INFORMATION_SCHEMA"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.bq_config = config.get('bigquery', {})
        self.projects = self.bq_config.get('projects', [])
        
        # Hostname patterns
        self.hostname_patterns = [
            'hostname', 'host_name', 'server_name', 'servername',
            'machine_name', 'computer_name', 'node_name', 'instance_name',
            'fqdn', 'host', 'hosts', 'server', 'servers', 'node', 'nodes'
        ]
        
        # Results
        self.tables_with_hosts = []
        self.hostname_columns_found = []
        self.unique_hostnames = set()
        
        # Stats
        self.scan_start_time = None
        self.tables_scanned = 0
        
        # Client pool
        self.client_managers = {}
        
        logger.info("=" * 80)
        logger.info("🚀 ULTRA-FAST BigQuery Scanner initialized")
        logger.info("=" * 80)
    
    def _get_client(self, project_id: str) -> bigquery.Client:
        """Get or create BigQuery client"""
        if project_id not in self.client_managers:
            self.client_managers[project_id] = BigQueryClientManager(project_id=project_id)
        return self.client_managers[project_id].get_client()
    
    async def scan_all_projects(self) -> List[Dict]:
        """ULTRA-FAST scan using INFORMATION_SCHEMA"""
        self.scan_start_time = time.time()
        all_data = []
        
        logger.info("\n" + "="*80)
        logger.info("📊 ULTRA-FAST HOSTNAME SCAN STARTING")
        logger.info("="*80)
        
        for project_id in self.projects:
            logger.info(f"\n📂 Scanning project: {project_id}")
            
            try:
                # Step 1: Find ALL hostname columns in ONE query
                hostname_tables = await self._find_all_hostname_columns_fast(project_id)
                
                if hostname_tables:
                    logger.info(f"  ✅ Found {len(hostname_tables)} tables with hostname columns")
                    
                    # Step 2: Sample data from these tables in parallel
                    with ThreadPoolExecutor(max_workers=20) as executor:
                        futures = []
                        
                        for table_info in hostname_tables:
                            future = executor.submit(
                                self._sample_hostname_data,
                                project_id,
                                table_info
                            )
                            futures.append(future)
                        
                        # Collect results
                        for future in as_completed(futures):
                            try:
                                result = future.result(timeout=2)
                                if result:
                                    all_data.append(result)
                            except Exception as e:
                                logger.debug(f"Sample error: {e}")
                
            except Exception as e:
                logger.error(f"❌ Error scanning project {project_id}: {e}")
        
        duration = time.time() - self.scan_start_time
        self._print_ultra_fast_summary(duration)
        
        return all_data
    
    async def _find_all_hostname_columns_fast(self, project_id: str) -> List[Dict]:
        """Find ALL hostname columns in ONE QUERY using INFORMATION_SCHEMA"""
        client = self._get_client(project_id)
        hostname_tables = []
        
        try:
            # Build the WHERE clause for hostname patterns
            column_patterns = []
            for pattern in self.hostname_patterns:
                column_patterns.append(f"LOWER(column_name) = '{pattern}'")
                column_patterns.append(f"LOWER(column_name) LIKE '%{pattern}%'")
            
            where_clause = " OR ".join(column_patterns)
            
            # SINGLE QUERY to find ALL hostname columns across ALL datasets
            query = f"""
            SELECT 
                table_catalog as project_id,
                table_schema as dataset_id,
                table_name,
                column_name,
                data_type
            FROM `{project_id}.INFORMATION_SCHEMA.COLUMNS`
            WHERE ({where_clause})
            ORDER BY table_schema, table_name
            """
            
            logger.info(f"  ⚡ Running INFORMATION_SCHEMA query...")
            start = time.time()
            
            query_job = client.query(query)
            results = list(query_job.result())
            
            query_time = time.time() - start
            logger.info(f"  ✅ Query completed in {query_time:.2f} seconds")
            
            # Group by table
            tables_dict = defaultdict(list)
            for row in results:
                table_key = f"{row.dataset_id}.{row.table_name}"
                tables_dict[table_key].append(row.column_name)
                
                # Track this hostname column
                self.hostname_columns_found.append(f"{table_key}.{row.column_name}")
            
            # Convert to list
            for table_name, columns in tables_dict.items():
                dataset_id, table_id = table_name.split('.', 1)
                hostname_tables.append({
                    'dataset_id': dataset_id,
                    'table_id': table_id,
                    'hostname_columns': columns
                })
                
                self.tables_with_hosts.append(f"{project_id}.{table_name}")
                self.tables_scanned += 1
            
            logger.info(f"  📊 Found {len(hostname_tables)} tables with hostname columns")
            
        except Exception as e:
            # Try region-specific INFORMATION_SCHEMA if the general one fails
            logger.warning(f"  ⚠️ INFORMATION_SCHEMA query failed, trying alternative method: {e}")
            hostname_tables = await self._fallback_scan(project_id)
        
        return hostname_tables
    
    async def _fallback_scan(self, project_id: str) -> List[Dict]:
        """Fallback: Check each dataset's INFORMATION_SCHEMA separately"""
        client = self._get_client(project_id)
        hostname_tables = []
        
        try:
            # Get all datasets first
            datasets = list(client.list_datasets())
            logger.info(f"  📁 Checking {len(datasets)} datasets individually...")
            
            for dataset in datasets:
                dataset_id = dataset.dataset_id
                
                try:
                    # Query this dataset's INFORMATION_SCHEMA
                    column_patterns = []
                    for pattern in self.hostname_patterns:
                        column_patterns.append(f"LOWER(column_name) = '{pattern}'")
                    
                    where_clause = " OR ".join(column_patterns)
                    
                    query = f"""
                    SELECT DISTINCT
                        table_name,
                        column_name
                    FROM `{project_id}.{dataset_id}.INFORMATION_SCHEMA.COLUMNS`
                    WHERE ({where_clause})
                    """
                    
                    query_job = client.query(query)
                    results = list(query_job.result(timeout=2))
                    
                    # Group by table
                    tables_in_dataset = defaultdict(list)
                    for row in results:
                        tables_in_dataset[row.table_name].append(row.column_name)
                    
                    # Add to results
                    for table_id, columns in tables_in_dataset.items():
                        hostname_tables.append({
                            'dataset_id': dataset_id,
                            'table_id': table_id,
                            'hostname_columns': columns
                        })
                        
                        self.tables_with_hosts.append(f"{project_id}.{dataset_id}.{table_id}")
                        self.hostname_columns_found.extend(
                            [f"{dataset_id}.{table_id}.{col}" for col in columns]
                        )
                    
                except Exception as e:
                    logger.debug(f"    Could not scan {dataset_id}: {e}")
            
        except Exception as e:
            logger.error(f"  ❌ Fallback scan failed: {e}")
        
        return hostname_tables
    
    def _sample_hostname_data(self, project_id: str, table_info: Dict) -> Optional[Dict]:
        """Sample hostname data from a specific table"""
        client = self._get_client(project_id)
        dataset_id = table_info['dataset_id']
        table_id = table_info['table_id']
        hostname_columns = table_info['hostname_columns'][:3]  # Max 3 columns
        
        full_table_id = f"{project_id}.{dataset_id}.{table_id}"
        
        try:
            # Build query for just hostname columns
            columns_str = ', '.join([f'`{col}`' for col in hostname_columns])
            
            # Use TABLESAMPLE for speed
            query = f"""
            SELECT {columns_str}
            FROM `{full_table_id}` TABLESAMPLE SYSTEM (1 PERCENT)
            WHERE {hostname_columns[0]} IS NOT NULL
            LIMIT 100
            """
            
            query_job = client.query(query)
            rows = list(query_job.result(timeout=2))
            
            # Collect samples
            hostname_samples = defaultdict(list)
            for row in rows:
                for col in hostname_columns:
                    value = row.get(col)
                    if value and self._looks_like_hostname(str(value)):
                        hostname_samples[col].append(str(value))
                        self.unique_hostnames.add(str(value))
            
            if hostname_samples:
                return {
                    'type': 'bigquery',
                    'source': full_table_id,
                    'has_hostnames': True,
                    'hostname_columns': hostname_columns,
                    'hostname_samples': dict(hostname_samples),
                    'tables': [{
                        'name': table_id,
                        'full_name': full_table_id,
                        'columns': {col: {'name': col, 'type': 'hostname'} 
                                  for col in hostname_columns}
                    }]
                }
            
        except Exception as e:
            logger.debug(f"Could not sample {full_table_id}: {e}")
        
        return None
    
    def _looks_like_hostname(self, value: str) -> bool:
        """Quick hostname validation"""
        if not value or len(value) > 253 or ' ' in value:
            return False
        
        # Basic patterns
        return bool(re.match(r'^[a-zA-Z0-9]([a-zA-Z0-9\-\.]*[a-zA-Z0-9])?$', value))
    
    def _print_ultra_fast_summary(self, duration: float):
        """Print summary"""
        print("\n" + "="*80)
        print("ULTRA-FAST SCAN COMPLETE")
        print("="*80)
        print(f"⏱️  Duration:              {duration:.2f} seconds")
        print(f"⚡ Tables/second:         {self.tables_scanned/duration:.1f}")
        print(f"📋 Tables with hostnames: {len(self.tables_with_hosts)}")
        print(f"🎯 Hostname columns:      {len(self.hostname_columns_found)}")
        print(f"🔤 Unique hostnames:      {len(self.unique_hostnames)}")
        print("="*80)
        
        if self.tables_with_hosts[:5]:
            print("\nSample tables with hostnames:")
            for table in self.tables_with_hosts[:5]:
                print(f"  • {table}")
    
    def get_training_data(self) -> Dict:
        """Get training data"""
        return {
            'hostnames': list(self.unique_hostnames),
            'tables_with_hosts': self.tables_with_hosts,
            'hostname_columns': self.hostname_columns_found
        }
    
    def get_statistics(self) -> Dict:
        """Get statistics"""
        duration = time.time() - self.scan_start_time if self.scan_start_time else 0
        
        return {
            'scan_duration_seconds': duration,
            'tables_scanned': self.tables_scanned,
            'tables_with_hosts': self.tables_with_hosts,
            'hostname_columns_found': len(self.hostname_columns_found),
            'unique_hostnames': len(self.unique_hostnames),
            'tables_per_second': self.tables_scanned / duration if duration > 0 else 0
        }