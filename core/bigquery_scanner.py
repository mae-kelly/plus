# core/bigquery_scanner.py
"""
BigQuery Scanner - Enhanced with multiple fallback approaches for handling arrays/NA
"""

import asyncio
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
import re
import os
import sys
import json
from collections import defaultdict

# Add parent directory to path to import BigQueryClientManager
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)

try:
    from google.cloud import bigquery
    from google.api_core import exceptions
    BIGQUERY_AVAILABLE = True
except ImportError:
    BIGQUERY_AVAILABLE = False
    logger.error("❌ Google Cloud BigQuery not installed!")
    logger.error("Run: pip install google-cloud-bigquery")

# Import the BigQueryClientManager
from core.bigquery_client_manager import BigQueryClientManager

class BigQueryScanner:
    """Scans BigQuery projects for infrastructure data with enhanced logging"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.bq_config = config.get('bigquery', {})
        
        # BigQuery settings
        self.projects = self.bq_config.get('projects', [])
        self.sample_percent = self.bq_config.get('sample_percent', 10)
        self.max_rows_per_table = self.bq_config.get('max_rows_per_table', 100000)
        
        # Filters
        self.datasets_filter = set(self.bq_config.get('datasets_filter', []))
        self.tables_filter = set(self.bq_config.get('tables_filter', []))
        
        # Statistics
        self.projects_scanned = []
        self.datasets_scanned = 0
        self.tables_scanned = 0
        self.rows_processed = 0
        self.tables_with_errors = []
        self.tables_with_hosts = []
        
        # Client managers for each project
        self.client_managers = {}
        
        # Track which approach worked for each table
        self.successful_approaches = {}
        
        # Validate setup
        if not BIGQUERY_AVAILABLE:
            raise ImportError("google-cloud-bigquery is required. Install it with: pip install google-cloud-bigquery")
        
        if not self.projects:
            raise ValueError("No BigQuery projects configured. Add project IDs to config.json under 'bigquery.projects'")
        
        logger.info("=" * 80)
        logger.info("🚀 BigQuery Scanner initialized")
        logger.info(f"📋 Projects to scan: {', '.join(self.projects)}")
        logger.info(f"📊 Sample percent: {self.sample_percent}%")
        logger.info(f"📈 Max rows per table: {self.max_rows_per_table:,}")
        logger.info("=" * 80)
    
    def _get_client(self, project_id: str) -> bigquery.Client:
        """Get or create BigQuery client using BigQueryClientManager"""
        if project_id not in self.client_managers:
            logger.info(f"🔑 Creating BigQuery client for project: {project_id}")
            self.client_managers[project_id] = BigQueryClientManager(project_id=project_id)
            
            # Test the connection
            if not self.client_managers[project_id].test_connection():
                raise ConnectionError(f"Failed to connect to BigQuery project: {project_id}")
            
            logger.info(f"✅ Connected to project: {project_id}")
        
        return self.client_managers[project_id].get_client()
    
    async def scan_all_projects(self) -> List[Dict]:
        """Scan all configured BigQuery projects"""
        all_data = []
        
        logger.info("\n" + "="*80)
        logger.info("📊 STARTING BIGQUERY SCAN")
        logger.info(f"🔍 Scanning {len(self.projects)} projects")
        logger.info("="*80)
        
        for project_idx, project_id in enumerate(self.projects, 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"📂 PROJECT {project_idx}/{len(self.projects)}: {project_id}")
            logger.info(f"{'='*60}")
            
            try:
                project_data = await self.scan_project(project_id)
                all_data.extend(project_data)
                self.projects_scanned.append(project_id)
                
                logger.info(f"✅ Completed scanning project: {project_id}")
                
            except Exception as e:
                logger.error(f"❌ Error scanning project {project_id}: {e}")
        
        # Final summary
        logger.info("\n" + "="*80)
        logger.info("📊 SCAN SUMMARY")
        logger.info(f"✅ Projects scanned: {len(self.projects_scanned)}/{len(self.projects)}")
        logger.info(f"📁 Datasets scanned: {self.datasets_scanned}")
        logger.info(f"📋 Tables scanned: {self.tables_scanned}")
        logger.info(f"📝 Rows processed: {self.rows_processed:,}")
        logger.info(f"🏠 Tables with potential hosts: {len(self.tables_with_hosts)}")
        logger.info(f"⚠️  Tables with errors: {len(self.tables_with_errors)}")
        
        # Report successful approaches
        if self.successful_approaches:
            logger.info("\n📊 Successful scanning approaches used:")
            approach_counts = defaultdict(int)
            for approach in self.successful_approaches.values():
                approach_counts[approach] += 1
            for approach, count in approach_counts.items():
                logger.info(f"   {approach}: {count} tables")
        
        logger.info("="*80)
        
        return all_data
    
    async def scan_project(self, project_id: str) -> List[Dict]:
        """Scan a single BigQuery project"""
        client = self._get_client(project_id)
        project_data = []
        
        try:
            logger.info(f"📁 Listing datasets in project {project_id}...")
            datasets = list(client.list_datasets(timeout=30))
            
            if not datasets:
                logger.warning(f"  ⚠️ No datasets found in project {project_id}")
                return project_data
            
            logger.info(f"  ✅ Found {len(datasets)} datasets")
            
            for dataset_idx, dataset_ref in enumerate(datasets, 1):
                dataset_id = dataset_ref.dataset_id
                
                if self.datasets_filter and dataset_id not in self.datasets_filter:
                    continue
                
                logger.info(f"\n  📁 DATASET {dataset_idx}/{len(datasets)}: {dataset_id}")
                
                dataset_data = await self.scan_dataset(client, project_id, dataset_id)
                if dataset_data:
                    project_data.extend(dataset_data)
                    self.datasets_scanned += 1
                    
        except Exception as e:
            logger.error(f"  ❌ Error listing datasets in {project_id}: {e}")
        
        return project_data
    
    async def scan_dataset(self, client: bigquery.Client, project_id: str, dataset_id: str) -> List[Dict]:
        """Scan all tables in a dataset"""
        dataset_data = []
        
        try:
            logger.info(f"    📋 Listing tables in {dataset_id}...")
            tables = list(client.list_tables(f"{project_id}.{dataset_id}", timeout=30))
            
            if not tables:
                logger.info(f"    ℹ️ No tables in dataset {dataset_id}")
                return dataset_data
            
            logger.info(f"    ✅ Found {len(tables)} tables")
            
            for table_idx, table_ref in enumerate(tables, 1):
                table_id = table_ref.table_id
                
                if self.tables_filter and table_id not in self.tables_filter:
                    continue
                
                logger.info(f"\n      📋 TABLE {table_idx}/{len(tables)}: {table_id}")
                
                try:
                    # Try multiple approaches until one works
                    table_data = await self.scan_table_with_fallback(client, project_id, dataset_id, table_id)
                    
                    if table_data:
                        dataset_data.append({
                            'type': 'bigquery',
                            'source': f'{project_id}.{dataset_id}.{table_id}',
                            'project': project_id,
                            'dataset': dataset_id,
                            'table': table_id,
                            'tables': [table_data]
                        })
                        self.tables_scanned += 1
                        logger.info(f"      ✅ Successfully scanned {table_id}")
                        
                except Exception as e:
                    logger.error(f"      ❌ All approaches failed for {table_id}: {e}")
                    self.tables_with_errors.append(f"{project_id}.{dataset_id}.{table_id}")
                    
        except Exception as e:
            logger.error(f"    ❌ Error scanning dataset {dataset_id}: {e}")
        
        return dataset_data
    
    async def scan_table_with_fallback(self, client: bigquery.Client, project_id: str, dataset_id: str, table_id: str) -> Optional[Dict]:
        """Try multiple scanning approaches until one works"""
        full_table_id = f"{project_id}.{dataset_id}.{table_id}"
        
        # List of approaches to try
        approaches = [
            ("Approach 1: Raw Iterator", self._scan_approach_1_raw_iterator),
            ("Approach 2: JSON Conversion in SQL", self._scan_approach_2_json_sql),
            ("Approach 3: Arrow Format", self._scan_approach_3_arrow),
            ("Approach 4: Pandas with Aggressive Cleaning", self._scan_approach_4_pandas_clean),
            ("Approach 5: Field by Field", self._scan_approach_5_field_by_field)
        ]
        
        for approach_name, approach_func in approaches:
            try:
                logger.info(f"        🔄 Trying {approach_name}...")
                result = await approach_func(client, project_id, dataset_id, table_id)
                if result:
                    logger.info(f"        ✅ {approach_name} succeeded!")
                    self.successful_approaches[full_table_id] = approach_name
                    return result
            except Exception as e:
                logger.debug(f"        ⚠️ {approach_name} failed: {str(e)[:100]}")
                continue
        
        # If all approaches fail, return None
        return None
    
    async def _scan_approach_1_raw_iterator(self, client: bigquery.Client, project_id: str, dataset_id: str, table_id: str) -> Optional[Dict]:
        """Approach 1: Use raw BigQuery iterator, no pandas"""
        full_table_id = f"{project_id}.{dataset_id}.{table_id}"
        
        table = client.get_table(full_table_id)
        if table.num_rows == 0:
            return None
        
        sample_size = min(int(table.num_rows * self.sample_percent / 100), self.max_rows_per_table)
        
        query = f"SELECT * FROM `{full_table_id}` LIMIT {sample_size}"
        query_job = client.query(query)
        
        rows = []
        columns_data = defaultdict(list)
        
        for bq_row in query_job.result():
            row_dict = {}
            for field in table.schema:
                field_name = field.name
                value = bq_row.get(field_name)
                
                # Convert everything to safe types
                if value is None:
                    clean_value = None
                elif isinstance(value, (list, tuple)):
                    clean_value = json.dumps(value)
                elif isinstance(value, dict):
                    clean_value = json.dumps(value)
                elif isinstance(value, bytes):
                    clean_value = value.decode('utf-8', errors='ignore')
                else:
                    clean_value = str(value) if value is not None else None
                
                row_dict[field_name] = clean_value
                columns_data[field_name].append(clean_value)
            
            rows.append(row_dict)
        
        columns = {}
        for field in table.schema:
            col_name = field.name
            columns[col_name] = {
                'name': col_name,
                'type': field.field_type,
                'mode': field.mode,
                'samples': columns_data[col_name][:100],
                'description': field.description,
                'statistics': self._analyze_column(columns_data[col_name]),
                'potential_type': self._infer_semantic_type(col_name, columns_data[col_name])
            }
        
        self.rows_processed += len(rows)
        
        return {
            'name': table_id,
            'full_name': full_table_id,
            'rows': rows,
            'columns': columns,
            'row_count': len(rows),
            'total_rows': table.num_rows
        }
    
    async def _scan_approach_2_json_sql(self, client: bigquery.Client, project_id: str, dataset_id: str, table_id: str) -> Optional[Dict]:
        """Approach 2: Convert complex types to JSON in SQL"""
        full_table_id = f"{project_id}.{dataset_id}.{table_id}"
        
        table = client.get_table(full_table_id)
        if table.num_rows == 0:
            return None
        
        sample_size = min(int(table.num_rows * self.sample_percent / 100), self.max_rows_per_table)
        
        # Build query that converts complex types to JSON
        select_parts = []
        for field in table.schema:
            if field.mode == 'REPEATED' or field.field_type in ['RECORD', 'STRUCT']:
                select_parts.append(f"TO_JSON_STRING({field.name}) AS {field.name}")
            else:
                select_parts.append(field.name)
        
        query = f"SELECT {', '.join(select_parts)} FROM `{full_table_id}` LIMIT {sample_size}"
        query_job = client.query(query)
        
        # Now everything is strings or simple types
        rows = []
        columns_data = defaultdict(list)
        
        for bq_row in query_job.result():
            row_dict = {}
            for field in table.schema:
                field_name = field.name
                value = bq_row.get(field_name)
                
                if value is None:
                    clean_value = None
                elif isinstance(value, bytes):
                    clean_value = value.decode('utf-8', errors='ignore')
                else:
                    clean_value = str(value) if value is not None else None
                
                row_dict[field_name] = clean_value
                columns_data[field_name].append(clean_value)
            
            rows.append(row_dict)
        
        columns = {}
        for field in table.schema:
            col_name = field.name
            columns[col_name] = {
                'name': col_name,
                'type': field.field_type,
                'mode': field.mode,
                'samples': columns_data[col_name][:100],
                'description': field.description,
                'statistics': self._analyze_column(columns_data[col_name]),
                'potential_type': self._infer_semantic_type(col_name, columns_data[col_name])
            }
        
        self.rows_processed += len(rows)
        
        return {
            'name': table_id,
            'full_name': full_table_id,
            'rows': rows,
            'columns': columns,
            'row_count': len(rows),
            'total_rows': table.num_rows
        }
    
    async def _scan_approach_3_arrow(self, client: bigquery.Client, project_id: str, dataset_id: str, table_id: str) -> Optional[Dict]:
        """Approach 3: Use Arrow format"""
        full_table_id = f"{project_id}.{dataset_id}.{table_id}"
        
        table = client.get_table(full_table_id)
        if table.num_rows == 0:
            return None
        
        sample_size = min(int(table.num_rows * self.sample_percent / 100), self.max_rows_per_table)
        
        query = f"SELECT * FROM `{full_table_id}` LIMIT {sample_size}"
        query_job = client.query(query)
        
        # Use Arrow format
        arrow_table = query_job.to_arrow()
        
        rows = []
        columns = {}
        
        # Process Arrow table
        for i in range(len(arrow_table)):
            row_dict = {}
            for col_name in arrow_table.column_names:
                value = arrow_table[col_name][i]
                
                if hasattr(value, 'as_py'):
                    py_value = value.as_py()
                    if py_value is None:
                        row_dict[col_name] = None
                    elif isinstance(py_value, (list, dict)):
                        row_dict[col_name] = json.dumps(py_value)
                    else:
                        row_dict[col_name] = py_value
                else:
                    row_dict[col_name] = str(value) if value is not None else None
            
            rows.append(row_dict)
        
        # Build columns
        for field in table.schema:
            col_name = field.name
            if col_name in arrow_table.column_names:
                col_data = []
                for val in arrow_table[col_name].to_pylist()[:100]:
                    if val is None:
                        col_data.append(None)
                    elif isinstance(val, (list, dict)):
                        col_data.append(json.dumps(val))
                    else:
                        col_data.append(val)
                
                columns[col_name] = {
                    'name': col_name,
                    'type': field.field_type,
                    'mode': field.mode,
                    'samples': col_data,
                    'description': field.description,
                    'statistics': self._analyze_column(col_data),
                    'potential_type': self._infer_semantic_type(col_name, col_data)
                }
        
        self.rows_processed += len(rows)
        
        return {
            'name': table_id,
            'full_name': full_table_id,
            'rows': rows,
            'columns': columns,
            'row_count': len(rows),
            'total_rows': table.num_rows
        }
    
    async def _scan_approach_4_pandas_clean(self, client: bigquery.Client, project_id: str, dataset_id: str, table_id: str) -> Optional[Dict]:
        """Approach 4: Use pandas with aggressive NA cleaning"""
        import pandas as pd
        import numpy as np
        
        full_table_id = f"{project_id}.{dataset_id}.{table_id}"
        
        table = client.get_table(full_table_id)
        if table.num_rows == 0:
            return None
        
        sample_size = min(int(table.num_rows * self.sample_percent / 100), self.max_rows_per_table)
        
        query = f"SELECT * FROM `{full_table_id}` LIMIT {sample_size}"
        query_job = client.query(query)
        
        # Get DataFrame
        df = query_job.to_dataframe()
        
        # Aggressively clean DataFrame
        for col in df.columns:
            # Convert column to object type first
            df[col] = df[col].astype('object')
            
            # Replace all NA-like values
            df[col] = df[col].where(pd.notnull(df[col]), None)
            
            # Convert arrays to strings
            for idx in df.index:
                val = df.at[idx, col]
                if isinstance(val, (list, np.ndarray)):
                    df.at[idx, col] = json.dumps(val.tolist() if hasattr(val, 'tolist') else list(val))
                elif isinstance(val, dict):
                    df.at[idx, col] = json.dumps(val)
                elif pd.isna(val):
                    df.at[idx, col] = None
        
        # Convert to records
        rows = df.where(pd.notnull(df), None).to_dict('records')
        
        # Build columns
        columns = {}
        for field in table.schema:
            col_name = field.name
            if col_name in df.columns:
                col_values = df[col_name].where(pd.notnull(df[col_name]), None).tolist()
            else:
                col_values = []
            
            columns[col_name] = {
                'name': col_name,
                'type': field.field_type,
                'mode': field.mode,
                'samples': col_values[:100],
                'description': field.description,
                'statistics': self._analyze_column(col_values),
                'potential_type': self._infer_semantic_type(col_name, col_values)
            }
        
        self.rows_processed += len(rows)
        
        return {
            'name': table_id,
            'full_name': full_table_id,
            'rows': rows,
            'columns': columns,
            'row_count': len(rows),
            'total_rows': table.num_rows
        }
    
    async def _scan_approach_5_field_by_field(self, client: bigquery.Client, project_id: str, dataset_id: str, table_id: str) -> Optional[Dict]:
        """Approach 5: Query each field separately to isolate problematic ones"""
        full_table_id = f"{project_id}.{dataset_id}.{table_id}"
        
        table = client.get_table(full_table_id)
        if table.num_rows == 0:
            return None
        
        sample_size = min(int(table.num_rows * self.sample_percent / 100), self.max_rows_per_table)
        
        # First get a row identifier or just row numbers
        query = f"SELECT * FROM `{full_table_id}` LIMIT 1"
        query_job = client.query(query)
        test_row = list(query_job.result())
        
        if not test_row:
            return None
        
        # Build complete result
        rows = []
        columns = {}
        
        # Get all data with safe field handling
        safe_fields = []
        for field in table.schema:
            if field.mode == 'REPEATED' or field.field_type in ['RECORD', 'STRUCT', 'ARRAY']:
                safe_fields.append(f"CAST(TO_JSON_STRING({field.name}) AS STRING) AS {field.name}")
            else:
                safe_fields.append(f"CAST({field.name} AS STRING) AS {field.name}")
        
        query = f"SELECT {', '.join(safe_fields)} FROM `{full_table_id}` LIMIT {sample_size}"
        query_job = client.query(query)
        
        columns_data = defaultdict(list)
        
        for bq_row in query_job.result():
            row_dict = {}
            for field in table.schema:
                field_name = field.name
                value = bq_row.get(field_name)
                
                # Everything is already a string or None
                row_dict[field_name] = value
                columns_data[field_name].append(value)
            
            rows.append(row_dict)
        
        # Build columns metadata
        for field in table.schema:
            col_name = field.name
            columns[col_name] = {
                'name': col_name,
                'type': field.field_type,
                'mode': field.mode,
                'samples': columns_data[col_name][:100],
                'description': field.description,
                'statistics': self._analyze_column(columns_data[col_name]),
                'potential_type': self._infer_semantic_type(col_name, columns_data[col_name])
            }
        
        self.rows_processed += len(rows)
        
        return {
            'name': table_id,
            'full_name': full_table_id,
            'rows': rows,
            'columns': columns,
            'row_count': len(rows),
            'total_rows': table.num_rows
        }
    
    def _check_for_hosts(self, table_data: Dict) -> bool:
        """Check if table likely contains host information"""
        columns = table_data.get('columns', {})
        
        for col_name, col_info in columns.items():
            if col_info.get('potential_type') in ['hostname', 'ip_address']:
                return True
        
        return False
    
    def _analyze_column(self, samples: List[Any]) -> Dict:
        """Analyze column samples for patterns"""
        if not samples:
            return {}
        
        # Filter out None values
        non_null = [s for s in samples if s is not None]
        
        if not non_null:
            return {'all_null': True}
        
        stats = {
            'count': len(samples),
            'unique': len(set(str(s) for s in non_null)),
            'null_count': len(samples) - len(non_null),
            'unique_ratio': len(set(str(s) for s in non_null)) / len(non_null) if non_null else 0
        }
        
        # Check for patterns (handle JSON strings)
        string_values = []
        for s in non_null:
            if isinstance(s, str):
                # Try to extract from JSON if it's a JSON string
                if s.startswith('[') or s.startswith('{'):
                    try:
                        parsed = json.loads(s)
                        if isinstance(parsed, list) and parsed:
                            string_values.append(str(parsed[0]))
                        else:
                            string_values.append(s)
                    except:
                        string_values.append(s)
                else:
                    string_values.append(s)
            else:
                string_values.append(str(s))
        
        # Check for IP addresses
        ip_pattern = r'^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$'
        ip_matches = sum(1 for s in string_values if re.match(ip_pattern, s))
        stats['ip_likelihood'] = ip_matches / len(string_values) if string_values else 0
        
        # Check for hostnames/FQDNs
        hostname_pattern = r'^[a-zA-Z0-9][a-zA-Z0-9\-\.]*[a-zA-Z0-9]$'
        hostname_matches = sum(1 for s in string_values 
                             if re.match(hostname_pattern, s) and '.' in s)
        stats['hostname_likelihood'] = hostname_matches / len(string_values) if string_values else 0
        
        return stats
    
    def _infer_semantic_type(self, column_name: str, samples: List[Any]) -> str:
        """Infer the semantic type of a column"""
        col_lower = column_name.lower()
        
        # Check column name patterns
        if any(pattern in col_lower for pattern in ['hostname', 'host_name', 'fqdn', 'servername', 'server_name']):
            return 'hostname'
        elif 'ip' in col_lower and any(pattern in col_lower for pattern in ['address', 'addr']):
            return 'ip_address'
        elif any(pattern in col_lower for pattern in ['environment', 'env']):
            return 'environment'
        elif any(pattern in col_lower for pattern in ['application', 'app', 'service']):
            return 'application'
        
        # Check sample data patterns
        if samples:
            non_null = []
            for s in samples[:100]:
                if s is not None:
                    if isinstance(s, str) and (s.startswith('[') or s.startswith('{')):
                        try:
                            parsed = json.loads(s)
                            if isinstance(parsed, list) and parsed:
                                non_null.append(str(parsed[0]))
                            else:
                                non_null.append(s)
                        except:
                            non_null.append(s)
                    else:
                        non_null.append(str(s) if s is not None else '')
            
            if non_null:
                # Check for IP pattern
                ip_matches = sum(1 for s in non_null 
                               if re.match(r'^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$', s))
                if ip_matches > len(non_null) * 0.8:
                    return 'ip_address'
                
                # Check for hostname/FQDN pattern
                hostname_matches = sum(1 for s in non_null 
                                     if re.match(r'^[a-zA-Z0-9][a-zA-Z0-9\-\.]*[a-zA-Z0-9]$', s)
                                     and '.' in s)
                if hostname_matches > len(non_null) * 0.6:
                    return 'hostname'
        
        return 'unknown'
    
    def get_statistics(self) -> Dict:
        """Get scanning statistics"""
        return {
            'projects_scanned': len(self.projects_scanned),
            'projects_list': self.projects_scanned,
            'datasets_scanned': self.datasets_scanned,
            'tables_scanned': self.tables_scanned,
            'rows_processed': self.rows_processed,
            'tables_with_hosts': self.tables_with_hosts,
            'tables_with_errors': self.tables_with_errors,
            'successful_approaches': self.successful_approaches
        }