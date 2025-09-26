# core/bigquery_scanner.py
"""
BigQuery Scanner - FAST hostname-focused scanning without pandas/numpy
"""

import asyncio
from typing import Dict, List, Any, Optional, Tuple, Set
import logging
from datetime import datetime
import re
import os
import sys
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logger = logging.getLogger(__name__)

try:
    from google.cloud import bigquery
    from google.api_core import exceptions
    BIGQUERY_AVAILABLE = True
except ImportError:
    BIGQUERY_AVAILABLE = False
    logger.error("❌ Google Cloud BigQuery not installed!")

from core.bigquery_client_manager import BigQueryClientManager

class BigQueryScanner:
    """FAST scanner that ONLY looks for hostname columns"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.bq_config = config.get('bigquery', {})
        self.projects = self.bq_config.get('projects', [])
        
        # Performance settings
        self.max_workers = self.bq_config.get('max_workers', 10)  # Parallel table scanning
        self.sample_size = 100  # Only get 100 samples per hostname column
        self.scan_timeout = 5  # 5 seconds max per table
        
        # Hostname patterns - ONLY look for these
        self.hostname_column_patterns = [
            r'.*hostname.*',
            r'.*host_name.*',
            r'.*servername.*',
            r'.*server_name.*',
            r'.*machine_name.*',
            r'.*computer_name.*',
            r'.*node_name.*',
            r'.*instance_name.*',
            r'.*fqdn.*',
            r'^host$',
            r'^hosts$',
            r'^server$',
            r'^servers$',
            r'^node$',
            r'^nodes$',
            r'^machine$',
            r'^machines$',
        ]
        
        # Results
        self.tables_with_hosts = []
        self.tables_without_hosts = []
        self.hostname_training_data = defaultdict(list)
        self.learned_patterns = []
        
        # Statistics
        self.projects_scanned = []
        self.datasets_scanned = 0
        self.tables_scanned = 0
        self.hostname_columns_found = 0
        self.unique_hostnames = set()
        self.scan_start_time = None
        self.scan_end_time = None
        
        # Client pool
        self.client_managers = {}
        
        if not BIGQUERY_AVAILABLE:
            raise ImportError("google-cloud-bigquery is required")
        
        if not self.projects:
            raise ValueError("No BigQuery projects configured")
        
        logger.info("=" * 80)
        logger.info("🚀 FAST BigQuery Hostname Scanner initialized")
        logger.info(f"📋 Projects to scan: {', '.join(self.projects)}")
        logger.info(f"⚡ Parallel workers: {self.max_workers}")
        logger.info("=" * 80)
    
    def _get_client(self, project_id: str) -> bigquery.Client:
        """Get or create BigQuery client"""
        if project_id not in self.client_managers:
            self.client_managers[project_id] = BigQueryClientManager(project_id=project_id)
        return self.client_managers[project_id].get_client()
    
    async def scan_all_projects(self) -> List[Dict]:
        """FAST scan - only look for hostname columns"""
        self.scan_start_time = time.time()
        all_data = []
        
        logger.info("\n" + "="*80)
        logger.info("📊 STARTING FAST HOSTNAME SCAN")
        logger.info("="*80)
        
        for project_idx, project_id in enumerate(self.projects, 1):
            logger.info(f"\n📂 PROJECT {project_idx}/{len(self.projects)}: {project_id}")
            
            try:
                project_data = await self.scan_project_fast(project_id)
                all_data.extend(project_data)
                self.projects_scanned.append(project_id)
            except Exception as e:
                logger.error(f"❌ Error scanning project {project_id}: {e}")
        
        self.scan_end_time = time.time()
        
        # Learn patterns from collected hostnames
        if self.hostname_training_data:
            self._learn_hostname_patterns()
        
        # Print summary
        self._print_fast_summary()
        
        return all_data
    
    async def scan_project_fast(self, project_id: str) -> List[Dict]:
        """Fast scan a project - parallel table scanning"""
        client = self._get_client(project_id)
        project_data = []
        
        try:
            # Get all datasets
            datasets = list(client.list_datasets(timeout=30))
            logger.info(f"  ✅ Found {len(datasets)} datasets")
            
            # Collect all tables
            all_tables = []
            for dataset_ref in datasets:
                dataset_id = dataset_ref.dataset_id
                tables = list(client.list_tables(f"{project_id}.{dataset_id}"))
                for table in tables:
                    all_tables.append((project_id, dataset_id, table.table_id))
                self.datasets_scanned += 1
            
            logger.info(f"  📋 Found {len(all_tables)} tables to scan")
            logger.info(f"  ⚡ Scanning with {self.max_workers} parallel workers...")
            
            # Process tables in parallel
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all scan jobs
                future_to_table = {
                    executor.submit(
                        self._fast_scan_table_for_hostnames,
                        client, project_id, dataset_id, table_id
                    ): (project_id, dataset_id, table_id)
                    for project_id, dataset_id, table_id in all_tables
                }
                
                # Process results as they complete
                completed = 0
                for future in as_completed(future_to_table):
                    table_info = future_to_table[future]
                    completed += 1
                    
                    try:
                        result = future.result(timeout=self.scan_timeout)
                        
                        if result:
                            if result.get('has_hostnames'):
                                project_data.append(result)
                                table_name = f"{table_info[0]}.{table_info[1]}.{table_info[2]}"
                                self.tables_with_hosts.append(table_name)
                                
                                # Collect training data
                                for col_name, samples in result.get('hostname_samples', {}).items():
                                    self.hostname_training_data[col_name].extend(samples)
                                    self.unique_hostnames.update(samples)
                                
                                logger.info(f"    ✅ [{completed}/{len(all_tables)}] {table_info[2]} - "
                                          f"FOUND {len(result.get('hostname_columns', []))} hostname columns")
                            else:
                                table_name = f"{table_info[0]}.{table_info[1]}.{table_info[2]}"
                                self.tables_without_hosts.append(table_name)
                        
                        self.tables_scanned += 1
                        
                        # Progress update every 10 tables
                        if completed % 10 == 0:
                            elapsed = time.time() - self.scan_start_time
                            rate = completed / elapsed if elapsed > 0 else 0
                            logger.info(f"    ⏱️  Progress: {completed}/{len(all_tables)} tables "
                                      f"({rate:.1f} tables/sec)")
                    
                    except Exception as e:
                        logger.debug(f"    ⚠️ Timeout/error scanning {table_info[2]}: {str(e)[:50]}")
                        self.tables_scanned += 1
            
        except Exception as e:
            logger.error(f"  ❌ Error scanning project: {e}")
        
        return project_data
    
    def _fast_scan_table_for_hostnames(self, client: bigquery.Client, 
                                       project_id: str, dataset_id: str, 
                                       table_id: str) -> Optional[Dict]:
        """Ultra-fast hostname column detection"""
        full_table_id = f"{project_id}.{dataset_id}.{table_id}"
        
        try:
            # Step 1: Get schema ONLY
            table = client.get_table(full_table_id)
            
            # Step 2: Find hostname columns in schema (FAST)
            hostname_columns = []
            for field in table.schema:
                col_lower = field.name.lower()
                
                # Check if column name matches hostname patterns
                for pattern in self.hostname_column_patterns:
                    if re.match(pattern, col_lower):
                        hostname_columns.append(field.name)
                        self.hostname_columns_found += 1
                        break
            
            # Step 3: If no hostname columns, return immediately
            if not hostname_columns:
                return {'has_hostnames': False}
            
            # Step 4: Sample ONLY hostname columns (FAST query)
            hostname_samples = {}
            
            # Limit to first 3 hostname columns to keep query fast
            columns_to_sample = hostname_columns[:3]
            
            # Build efficient query
            select_parts = []
            for col in columns_to_sample:
                select_parts.append(f"`{col}`")
            
            query = f"""
            SELECT DISTINCT {', '.join(select_parts)}
            FROM `{full_table_id}`
            WHERE {columns_to_sample[0]} IS NOT NULL
            LIMIT {self.sample_size}
            """
            
            # Execute query with timeout
            query_job = client.query(query)
            query_job.result(timeout=3)  # 3 second timeout
            
            # Collect samples
            for row in query_job:
                for col in columns_to_sample:
                    value = row.get(col)
                    if value and self._looks_like_hostname(str(value)):
                        if col not in hostname_samples:
                            hostname_samples[col] = []
                        hostname_samples[col].append(str(value))
            
            # Limit samples
            for col in hostname_samples:
                hostname_samples[col] = list(set(hostname_samples[col][:50]))
            
            return {
                'type': 'bigquery',
                'source': full_table_id,
                'has_hostnames': True,
                'hostname_columns': hostname_columns,
                'hostname_samples': hostname_samples,
                'tables': [{
                    'name': table_id,
                    'full_name': full_table_id,
                    'columns': {col: {'name': col, 'type': 'hostname'} for col in hostname_columns},
                    'row_count': table.num_rows
                }]
            }
            
        except Exception as e:
            # Silent fail - just log debug
            logger.debug(f"Could not scan {table_id}: {str(e)[:100]}")
            return None
    
    def _looks_like_hostname(self, value: str) -> bool:
        """Quick check if value looks like a hostname"""
        if not value or len(value) > 253:
            return False
        
        # Quick rejections
        if ' ' in value or '\n' in value or '\t' in value:
            return False
        
        # Basic hostname pattern
        if re.match(r'^[a-zA-Z0-9]([a-zA-Z0-9\-\.]*[a-zA-Z0-9])?$', value):
            return True
        
        # IP address pattern
        if re.match(r'^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$', value):
            return True
        
        return False
    
    def _learn_hostname_patterns(self):
        """Learn patterns from collected hostname data"""
        logger.info("\n🧠 Learning hostname patterns from collected data...")
        
        all_hostnames = list(self.unique_hostnames)[:1000]  # Sample for pattern learning
        
        patterns = defaultdict(int)
        
        for hostname in all_hostnames:
            # Hyphenated patterns
            if '-' in hostname:
                parts = hostname.split('-')
                patterns[f'hyphenated_{len(parts)}_parts'] += 1
                
                # Check for environment prefixes
                if parts[0].lower() in ['prod', 'dev', 'test', 'qa', 'uat', 'stg']:
                    patterns[f'env_prefix_{parts[0].lower()}'] += 1
            
            # FQDN patterns
            if '.' in hostname:
                parts = hostname.split('.')
                patterns[f'fqdn_{len(parts)}_levels'] += 1
            
            # Cloud provider patterns
            if hostname.startswith('i-') and len(hostname) > 10:
                patterns['aws_instance_pattern'] += 1
            elif re.match(r'^[a-z]+-[a-z0-9]+-[a-z0-9]+', hostname):
                patterns['gcp_instance_pattern'] += 1
            
            # Numbered patterns
            if re.search(r'\d+$', hostname):
                patterns['numbered_suffix'] += 1
            
            # Length patterns
            if len(hostname) < 10:
                patterns['short_hostname'] += 1
            elif len(hostname) > 30:
                patterns['long_hostname'] += 1
        
        # Store learned patterns
        self.learned_patterns = [
            {'pattern': k, 'count': v, 'frequency': v/len(all_hostnames)}
            for k, v in patterns.items()
            if v >= 2  # Pattern must appear at least twice
        ]
        
        # Sort by frequency
        self.learned_patterns.sort(key=lambda x: x['frequency'], reverse=True)
        
        logger.info(f"  ✅ Learned {len(self.learned_patterns)} hostname patterns")
        
        # Show top patterns
        for pattern in self.learned_patterns[:5]:
            logger.info(f"    • {pattern['pattern']}: {pattern['frequency']:.1%} "
                       f"({pattern['count']} occurrences)")
    
    def _print_fast_summary(self):
        """Print scan summary"""
        duration = self.scan_end_time - self.scan_start_time if self.scan_end_time else 0
        tables_per_sec = self.tables_scanned / duration if duration > 0 else 0
        
        print("\n" + "="*80)
        print("FAST HOSTNAME SCAN COMPLETE")
        print("="*80)
        print(f"⏱️  Duration:                 {duration:.1f} seconds")
        print(f"⚡ Speed:                    {tables_per_sec:.1f} tables/second")
        print("-"*80)
        print(f"📊 Projects Scanned:         {len(self.projects_scanned)}")
        print(f"📁 Datasets Scanned:         {self.datasets_scanned}")
        print(f"📋 Tables Scanned:           {self.tables_scanned}")
        print("-"*80)
        print(f"✅ Tables WITH hostnames:    {len(self.tables_with_hosts)}")
        print(f"❌ Tables WITHOUT hostnames: {len(self.tables_without_hosts)}")
        print(f"🎯 Hostname columns found:   {self.hostname_columns_found}")
        print(f"🔤 Unique hostnames:         {len(self.unique_hostnames)}")
        print(f"🧠 Patterns learned:         {len(self.learned_patterns)}")
        print("="*80)
        
        # Show sample tables with hostnames
        if self.tables_with_hosts:
            print("\nSample Tables with Hostnames:")
            for table in self.tables_with_hosts[:5]:
                print(f"  • {table}")
        
        # Show sample hostnames for training
        if self.unique_hostnames:
            print("\nSample Hostnames for Training:")
            for hostname in list(self.unique_hostnames)[:10]:
                print(f"  • {hostname}")
    
    def get_training_data(self) -> Dict:
        """Get hostname training data for ML models"""
        return {
            'hostnames': list(self.unique_hostnames),
            'hostname_columns': list(self.hostname_training_data.keys()),
            'patterns': self.learned_patterns,
            'tables_with_hosts': self.tables_with_hosts,
            'column_samples': dict(self.hostname_training_data)
        }
    
    def export_results(self, filepath: str = 'hostname_scan_results.json'):
        """Export scan results"""
        results = {
            'scan_timestamp': datetime.now().isoformat(),
            'duration_seconds': self.scan_end_time - self.scan_start_time if self.scan_end_time else 0,
            'statistics': {
                'projects_scanned': len(self.projects_scanned),
                'datasets_scanned': self.datasets_scanned,
                'tables_scanned': self.tables_scanned,
                'tables_with_hostnames': len(self.tables_with_hosts),
                'tables_without_hostnames': len(self.tables_without_hosts),
                'hostname_columns_found': self.hostname_columns_found,
                'unique_hostnames': len(self.unique_hostnames),
                'patterns_learned': len(self.learned_patterns)
            },
            'tables_with_hosts': self.tables_with_hosts[:100],  # Limit for file size
            'learned_patterns': self.learned_patterns[:20],
            'sample_hostnames': list(self.unique_hostnames)[:100]
        }
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"📁 Results exported to {filepath}")
        
        return results
    
    def get_statistics(self) -> Dict:
        """Get scanning statistics"""
        duration = self.scan_end_time - self.scan_start_time if self.scan_end_time else 0
        
        return {
            'projects_scanned': len(self.projects_scanned),
            'projects_list': self.projects_scanned,
            'datasets_scanned': self.datasets_scanned,
            'tables_scanned': self.tables_scanned,
            'tables_with_hosts': self.tables_with_hosts,
            'tables_without_hosts': self.tables_without_hosts[:100],  # Limit list size
            'hostname_columns_found': self.hostname_columns_found,
            'unique_hostnames_count': len(self.unique_hostnames),
            'patterns_learned': len(self.learned_patterns),
            'scan_duration_seconds': duration,
            'tables_per_second': self.tables_scanned / duration if duration > 0 else 0
        }