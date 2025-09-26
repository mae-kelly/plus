# core/bigquery_scanner.py
"""
BigQuery Scanner - Enhanced with detailed logging and NA handling
"""

import asyncio
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
import re
import os
import sys
import pandas as pd
import numpy as np
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
                # Get project info
                if project_id in self.client_managers:
                    project_info = self.client_managers[project_id].get_project_info()
                    logger.info(f"✅ Connected to: {project_info.get('friendly_name', project_id)}")
                
                project_data = await self.scan_project(project_id)
                all_data.extend(project_data)
                self.projects_scanned.append(project_id)
                
                logger.info(f"✅ Completed scanning project: {project_id}")
                
            except exceptions.Forbidden as e:
                logger.error(f"❌ Permission denied for project {project_id}")
                logger.error(f"   Error: {e}")
                logger.error(f"   Make sure your credentials have 'BigQuery Data Viewer' role")
                
            except ConnectionError as e:
                logger.error(f"❌ Connection failed for project {project_id}")
                logger.error(f"   Error: {e}")
                
            except Exception as e:
                logger.error(f"❌ Unexpected error scanning project {project_id}")
                logger.error(f"   Error type: {type(e).__name__}")
                logger.error(f"   Error: {e}")
        
        # Final summary
        logger.info("\n" + "="*80)
        logger.info("📊 SCAN SUMMARY")
        logger.info(f"✅ Projects scanned: {len(self.projects_scanned)}/{len(self.projects)}")
        logger.info(f"📁 Datasets scanned: {self.datasets_scanned}")
        logger.info(f"📋 Tables scanned: {self.tables_scanned}")
        logger.info(f"📝 Rows processed: {self.rows_processed:,}")
        logger.info(f"🏠 Tables with potential hosts: {len(self.tables_with_hosts)}")
        logger.info(f"⚠️  Tables with errors: {len(self.tables_with_errors)}")
        logger.info("="*80)
        
        if not self.projects_scanned:
            logger.error("❌ No projects were successfully scanned!")
            logger.error("Please check:")
            logger.error("1. Your project IDs in config.json are correct")
            logger.error("2. Your credentials have BigQuery access")
            logger.error("3. The authentication is properly configured")
        
        return all_data
    
    async def scan_project(self, project_id: str) -> List[Dict]:
        """Scan a single BigQuery project"""
        client = self._get_client(project_id)
        project_data = []
        
        # List datasets
        try:
            logger.info(f"📁 Listing datasets in project {project_id}...")
            datasets = list(client.list_datasets(timeout=30))
            
            if not datasets:
                logger.warning(f"  ⚠️ No datasets found in project {project_id}")
                return project_data
            
            logger.info(f"  ✅ Found {len(datasets)} datasets")
            
            for dataset_idx, dataset_ref in enumerate(datasets, 1):
                dataset_id = dataset_ref.dataset_id
                
                # Apply filter
                if self.datasets_filter and dataset_id not in self.datasets_filter:
                    logger.debug(f"  Skipping dataset {dataset_id} (not in filter)")
                    continue
                
                logger.info(f"\n  {'='*50}")
                logger.info(f"  📁 DATASET {dataset_idx}/{len(datasets)}: {dataset_id}")
                logger.info(f"  {'='*50}")
                
                dataset_data = await self.scan_dataset(client, project_id, dataset_id)
                if dataset_data:
                    project_data.extend(dataset_data)
                    self.datasets_scanned += 1
                    
        except exceptions.Forbidden as e:
            logger.error(f"  ❌ Permission denied listing datasets in {project_id}")
            logger.error(f"     Need 'bigquery.datasets.get' permission")
            
        except Exception as e:
            logger.error(f"  ❌ Error listing datasets in {project_id}: {e}")
        
        return project_data
    
    async def scan_dataset(self, client: bigquery.Client, project_id: str, dataset_id: str) -> List[Dict]:
        """Scan all tables in a dataset"""
        dataset_data = []
        
        try:
            # List tables
            logger.info(f"    📋 Listing tables in {dataset_id}...")
            tables = list(client.list_tables(f"{project_id}.{dataset_id}", timeout=30))
            
            if not tables:
                logger.info(f"    ℹ️ No tables in dataset {dataset_id}")
                return dataset_data
            
            logger.info(f"    ✅ Found {len(tables)} tables")
            
            # Process each table
            for table_idx, table_ref in enumerate(tables, 1):
                table_id = table_ref.table_id
                
                # Apply filter
                if self.tables_filter and table_id not in self.tables_filter:
                    logger.debug(f"    Skipping table {table_id} (not in filter)")
                    continue
                
                logger.info(f"\n      {'='*40}")
                logger.info(f"      📋 TABLE {table_idx}/{len(tables)}: {table_id}")
                logger.info(f"      {'='*40}")
                
                try:
                    table_data = await self.scan_table(client, project_id, dataset_id, table_id)
                    if table_data:
                        # Check if table has potential hosts
                        has_hosts = self._check_for_hosts(table_data)
                        if has_hosts:
                            logger.info(f"      🏠 Found potential hosts in {table_id}")
                            self.tables_with_hosts.append(f"{project_id}.{dataset_id}.{table_id}")
                        
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
                    error_msg = str(e)
                    logger.error(f"      ❌ Error scanning table {table_id}")
                    logger.error(f"         Error: {error_msg}")
                    self.tables_with_errors.append(f"{project_id}.{dataset_id}.{table_id}")
                    
                    # Handle specific errors
                    if "ambiguous" in error_msg.lower():
                        logger.info(f"      ℹ️ Retrying with NA-safe method...")
                        try:
                            table_data = await self._scan_table_safe(
                                client, project_id, dataset_id, table_id
                            )
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
                                logger.info(f"      ✅ Successfully scanned {table_id} with safe method")
                        except Exception as e2:
                            logger.error(f"      ❌ Retry failed: {e2}")
                    
        except exceptions.Forbidden as e:
            logger.error(f"    ❌ Permission denied listing tables in {dataset_id}")
            
        except Exception as e:
            logger.error(f"    ❌ Error scanning dataset {dataset_id}: {e}")
        
        return dataset_data
    
    async def scan_table(self, client: bigquery.Client, project_id: str, dataset_id: str, table_id: str) -> Optional[Dict]:
        """Scan a BigQuery table and extract data"""
        full_table_id = f"{project_id}.{dataset_id}.{table_id}"
        
        try:
            # Get table metadata
            table = client.get_table(full_table_id)
            
            if table.num_rows == 0:
                logger.info(f"        ℹ️ Table {table_id} is empty")
                return None
            
            logger.info(f"        📊 Table stats:")
            logger.info(f"           Rows: {table.num_rows:,}")
            logger.info(f"           Size: {table.num_bytes/1024/1024:.1f} MB")
            logger.info(f"           Columns: {len(table.schema)}")
            
            # Log column names to help identify hostname columns
            column_names = [field.name for field in table.schema]
            logger.info(f"        📋 Columns: {', '.join(column_names[:10])}")
            if len(column_names) > 10:
                logger.info(f"           ... and {len(column_names)-10} more columns")
            
            # Determine how many rows to sample
            sample_size = min(
                int(table.num_rows * self.sample_percent / 100),
                self.max_rows_per_table
            )
            
            logger.info(f"        🎯 Sampling {sample_size:,} rows ({self.sample_percent}%)")
            
            # Build query
            if sample_size < table.num_rows:
                query = f"""
                SELECT *
                FROM `{full_table_id}`
                TABLESAMPLE SYSTEM ({min(self.sample_percent, 100)} PERCENT)
                LIMIT {sample_size}
                """
            else:
                query = f"""
                SELECT *
                FROM `{full_table_id}`
                LIMIT {sample_size}
                """
            
            logger.info(f"        ⏳ Executing query...")
            
            # Execute query
            query_job = client.query(query)
            
            # Get results as iterator of Row objects to avoid DataFrame issues
            rows_iterator = query_job.result()
            
            # Convert to list of dictionaries
            rows = []
            columns_data = defaultdict(list)
            
            for row in rows_iterator:
                row_dict = {}
                for field in table.schema:
                    field_name = field.name
                    value = row.get(field_name)
                    
                    # Clean the value - handle all types safely
                    cleaned_value = self._clean_value(value)
                    
                    row_dict[field_name] = cleaned_value
                    columns_data[field_name].append(cleaned_value)
                
                rows.append(row_dict)
            
            if not rows:
                logger.warning(f"        ⚠️ No rows returned from {table_id}")
                return None
            
            logger.info(f"        ✅ Retrieved {len(rows)} rows")
            
            # Build columns metadata
            columns = {}
            
            for field in table.schema:
                col_name = field.name
                col_values = columns_data.get(col_name, [])
                
                columns[col_name] = {
                    'name': col_name,
                    'type': field.field_type,
                    'mode': field.mode,
                    'samples': col_values[:100],  # Keep first 100 samples
                    'description': field.description
                }
                
                # Analyze column for discovery patterns
                columns[col_name]['statistics'] = self._analyze_column(col_values)
                columns[col_name]['potential_type'] = self._infer_semantic_type(col_name, col_values)
                
                # Log interesting columns
                if columns[col_name]['potential_type'] in ['hostname', 'ip_address']:
                    logger.info(f"        🎯 Found {columns[col_name]['potential_type']} column: {col_name}")
            
            self.rows_processed += len(rows)
            
            return {
                'name': table_id,
                'full_name': full_table_id,
                'rows': rows,
                'columns': columns,
                'row_count': len(rows),
                'total_rows': table.num_rows,
                'created': str(table.created) if table.created else None,
                'modified': str(table.modified) if table.modified else None,
                'size_bytes': table.num_bytes,
                'size_mb': round(table.num_bytes / 1024 / 1024, 2),
                'description': table.description
            }
            
        except Exception as e:
            logger.error(f"        ❌ Error scanning table {full_table_id}")
            logger.error(f"           Error type: {type(e).__name__}")
            logger.error(f"           Error: {e}")
            
            # If it's the specific NA error, try alternative method
            if "ambiguous" in str(e).lower():
                logger.info(f"        ℹ️ Retrying with NA-safe method...")
                return await self._scan_table_safe(client, project_id, dataset_id, table_id)
            else:
                raise
    
    def _clean_value(self, value):
        """Clean a single value, handling all types safely"""
        # Handle None
        if value is None:
            return None
        
        # Handle pandas NA or numpy nan
        try:
            # Check if it's a scalar that might be NA
            if hasattr(value, '__len__') and not isinstance(value, (str, bytes)):
                # It's array-like, convert to list
                if hasattr(value, 'tolist'):
                    return value.tolist()
                else:
                    return list(value)
            
            # For scalar values, check if it's NA without using ambiguous operations
            # Use pandas isna in a safe way
            if isinstance(value, float) and np.isnan(value):
                return None
        except (TypeError, ValueError):
            # If checking for NA fails, it's probably a valid value
            pass
        
        # Handle numpy types
        if hasattr(value, 'item'):
            try:
                return value.item()
            except:
                pass
        
        # Handle bytes
        if isinstance(value, bytes):
            return value.decode('utf-8', errors='ignore')
        
        # Return as-is if no special handling needed
        return value
    
    async def _scan_table_safe(self, client: bigquery.Client, project_id: str, dataset_id: str, table_id: str) -> Optional[Dict]:
        """Safe table scanning method that avoids pandas DataFrame and NA issues"""
        full_table_id = f"{project_id}.{dataset_id}.{table_id}"
        
        try:
            table = client.get_table(full_table_id)
            
            if table.num_rows == 0:
                return None
            
            # Use smaller sample for problematic tables
            sample_size = min(500, table.num_rows)
            
            query = f"""
            SELECT *
            FROM `{full_table_id}`
            LIMIT {sample_size}
            """
            
            query_job = client.query(query)
            rows_iterator = query_job.result()
            
            # Process without pandas
            rows = []
            columns = defaultdict(lambda: {
                'samples': [],
                'name': '',
                'type': '',
                'mode': '',
                'description': ''
            })
            
            for row in rows_iterator:
                row_dict = {}
                for field in table.schema:
                    field_name = field.name
                    value = row.get(field_name)
                    
                    # Simple cleaning without pandas
                    if value is None:
                        cleaned_value = None
                    elif isinstance(value, (list, tuple)):
                        cleaned_value = list(value)
                    elif isinstance(value, bytes):
                        cleaned_value = value.decode('utf-8', errors='ignore')
                    elif isinstance(value, float) and np.isnan(value):
                        cleaned_value = None
                    else:
                        cleaned_value = value
                    
                    row_dict[field_name] = cleaned_value
                    
                    # Store column info
                    if not columns[field_name]['name']:
                        columns[field_name]['name'] = field_name
                        columns[field_name]['type'] = field.field_type
                        columns[field_name]['mode'] = field.mode
                        columns[field_name]['description'] = field.description
                    
                    # Store samples
                    if len(columns[field_name]['samples']) < 100 and cleaned_value is not None:
                        columns[field_name]['samples'].append(cleaned_value)
                
                rows.append(row_dict)
            
            # Analyze columns
            for col_name, col_info in columns.items():
                col_info['statistics'] = self._analyze_column(col_info['samples'])
                col_info['potential_type'] = self._infer_semantic_type(col_name, col_info['samples'])
            
            self.rows_processed += len(rows)
            
            return {
                'name': table_id,
                'full_name': full_table_id,
                'rows': rows,
                'columns': dict(columns),
                'row_count': len(rows),
                'total_rows': table.num_rows,
                'created': str(table.created) if table.created else None,
                'modified': str(table.modified) if table.modified else None,
                'size_bytes': table.num_bytes,
                'size_mb': round(table.num_bytes / 1024 / 1024, 2),
                'description': table.description
            }
            
        except Exception as e:
            logger.error(f"Safe scan also failed for {full_table_id}: {e}")
            return None
    
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
        
        # Check for patterns
        if non_null:
            # Check for IP addresses
            ip_pattern = r'^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$'
            ip_matches = sum(1 for s in non_null if re.match(ip_pattern, str(s)))
            stats['ip_likelihood'] = ip_matches / len(non_null) if non_null else 0
            
            # Check for hostnames/FQDNs
            hostname_pattern = r'^[a-zA-Z0-9][a-zA-Z0-9\-\.]*[a-zA-Z0-9]$'
            hostname_matches = sum(1 for s in non_null if re.match(hostname_pattern, str(s)) and '.' in str(s))
            stats['hostname_likelihood'] = hostname_matches / len(non_null) if non_null else 0
        
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
        
        # Check sample data patterns if column name doesn't match
        if samples:
            non_null = [s for s in samples[:100] if s is not None]
            if non_null:
                # Check for IP pattern
                ip_matches = sum(1 for s in non_null 
                                if re.match(r'^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$', str(s)))
                if ip_matches > len(non_null) * 0.8:
                    return 'ip_address'
                
                # Check for hostname/FQDN pattern
                hostname_matches = sum(1 for s in non_null 
                                     if re.match(r'^[a-zA-Z0-9][a-zA-Z0-9\-\.]*[a-zA-Z0-9]$', str(s))
                                     and '.' in str(s))
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
            'tables_with_errors': self.tables_with_errors
        }