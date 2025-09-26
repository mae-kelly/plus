"""
BigQuery Scanner - Uses BigQueryClientManager for authentication
"""

import asyncio
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
import re
import os
import sys

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
    """Scans BigQuery projects for infrastructure data using BigQueryClientManager for auth"""
    
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
        
        # Client managers for each project
        self.client_managers = {}
        
        # Validate setup
        if not BIGQUERY_AVAILABLE:
            raise ImportError("google-cloud-bigquery is required. Install it with: pip install google-cloud-bigquery")
        
        if not self.projects:
            raise ValueError("No BigQuery projects configured. Add project IDs to config.json under 'bigquery.projects'")
    
    def _get_client(self, project_id: str) -> bigquery.Client:
        """Get or create BigQuery client using BigQueryClientManager"""
        if project_id not in self.client_managers:
            logger.info(f"Creating BigQuery client for project: {project_id}")
            self.client_managers[project_id] = BigQueryClientManager(project_id=project_id)
            
            # Test the connection
            if not self.client_managers[project_id].test_connection():
                raise ConnectionError(f"Failed to connect to BigQuery project: {project_id}")
        
        return self.client_managers[project_id].get_client()
    
    async def scan_all_projects(self) -> List[Dict]:
        """Scan all configured BigQuery projects"""
        all_data = []
        
        logger.info(f"Starting scan of {len(self.projects)} BigQuery projects")
        logger.info(f"Projects to scan: {', '.join(self.projects)}")
        
        for project_id in self.projects:
            logger.info(f"📊 Scanning BigQuery project: {project_id}")
            
            try:
                # Get project info
                if project_id in self.client_managers:
                    project_info = self.client_managers[project_id].get_project_info()
                    logger.info(f"Connected to: {project_info.get('friendly_name', project_id)}")
                
                project_data = await self.scan_project(project_id)
                all_data.extend(project_data)
                self.projects_scanned.append(project_id)
                
            except exceptions.Forbidden as e:
                logger.error(f"❌ Permission denied for project {project_id}")
                logger.error(f"   Error: {e}")
                logger.error(f"   Make sure your credentials have 'BigQuery Data Viewer' role")
                logger.error(f"   Project: {project_id}")
                logger.error(f"   Required roles: roles/bigquery.dataViewer or roles/bigquery.admin")
                
            except ConnectionError as e:
                logger.error(f"❌ Connection failed for project {project_id}")
                logger.error(f"   Error: {e}")
                logger.error(f"   Check that the project ID is correct and you have access")
                
            except Exception as e:
                logger.error(f"❌ Unexpected error scanning project {project_id}")
                logger.error(f"   Error type: {type(e).__name__}")
                logger.error(f"   Error: {e}")
        
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
            logger.info(f"  Listing datasets in project {project_id}...")
            datasets = list(client.list_datasets(timeout=30))
            
            if not datasets:
                logger.warning(f"  ⚠️ No datasets found in project {project_id}")
                logger.info(f"     This might be normal if the project has no BigQuery datasets")
                return project_data
            
            logger.info(f"  ✅ Found {len(datasets)} datasets in {project_id}")
            
            for dataset_ref in datasets:
                dataset_id = dataset_ref.dataset_id
                
                # Apply filter
                if self.datasets_filter and dataset_id not in self.datasets_filter:
                    logger.debug(f"  Skipping dataset {dataset_id} (not in filter)")
                    continue
                
                logger.info(f"  📁 Scanning dataset: {dataset_id}")
                
                dataset_data = await self.scan_dataset(client, project_id, dataset_id)
                if dataset_data:
                    project_data.extend(dataset_data)
                    self.datasets_scanned += 1
                    
        except exceptions.Forbidden as e:
            logger.error(f"  ❌ Permission denied listing datasets in {project_id}")
            logger.error(f"     Need 'bigquery.datasets.get' permission")
            raise
            
        except Exception as e:
            logger.error(f"  ❌ Error listing datasets in {project_id}: {e}")
            raise
        
        return project_data
    
    async def scan_dataset(self, client: bigquery.Client, project_id: str, dataset_id: str) -> List[Dict]:
        """Scan all tables in a dataset"""
        dataset_data = []
        
        try:
            # List tables
            logger.info(f"    Listing tables in {project_id}.{dataset_id}...")
            tables = list(client.list_tables(f"{project_id}.{dataset_id}", timeout=30))
            
            if not tables:
                logger.debug(f"    No tables found in dataset {dataset_id}")
                return dataset_data
            
            logger.info(f"    Found {len(tables)} tables in {dataset_id}")
            
            # Process each table
            for table_ref in tables:
                table_id = table_ref.table_id
                
                # Apply filter
                if self.tables_filter and table_id not in self.tables_filter:
                    logger.debug(f"    Skipping table {table_id} (not in filter)")
                    continue
                
                logger.info(f"      📋 Scanning table: {table_id}")
                
                try:
                    table_data = await self.scan_table(client, project_id, dataset_id, table_id)
                    if table_data:
                        dataset_data.append({
                            'type': 'bigquery',
                            'source': f'{project_id}.{dataset_id}.{table_id}',
                            'project': project_id,
                            'dataset': dataset_id,
                            'table': table_id,
                            'tables': [table_data]  # Compatible with discovery format
                        })
                        self.tables_scanned += 1
                        
                except Exception as e:
                    logger.error(f"      ❌ Error scanning table {table_id}: {e}")
                    # Continue with next table instead of failing completely
                    continue
                    
        except exceptions.Forbidden as e:
            logger.error(f"    ❌ Permission denied listing tables in {dataset_id}")
            logger.error(f"       Need 'bigquery.tables.list' permission")
            
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
                logger.debug(f"        Table {table_id} is empty")
                return None
            
            logger.info(f"        Table size: {table.num_rows:,} rows, {table.num_bytes/1024/1024:.1f} MB")
            
            # Determine how many rows to sample
            sample_size = min(
                int(table.num_rows * self.sample_percent / 100),
                self.max_rows_per_table
            )
            
            logger.info(f"        Sampling {sample_size:,} rows ({self.sample_percent}% or max {self.max_rows_per_table:,})")
            
            # Build query
            if sample_size < table.num_rows:
                # Use TABLESAMPLE for efficient sampling of large tables
                query = f"""
                SELECT *
                FROM `{full_table_id}`
                TABLESAMPLE SYSTEM ({min(self.sample_percent, 100)} PERCENT)
                LIMIT {sample_size}
                """
            else:
                # Get all rows for small tables
                query = f"""
                SELECT *
                FROM `{full_table_id}`
                """
            
            logger.debug(f"        Executing query...")
            
            # Execute query with timeout
            query_job = client.query(query)
            rows = list(query_job.result(timeout=60))
            
            if not rows:
                logger.warning(f"        No rows returned from {table_id}")
                return None
            
            logger.info(f"        Retrieved {len(rows)} rows")
            
            # Convert to standard format
            rows_list = []
            columns = {}
            
            # Process rows
            for row in rows:
                row_dict = {}
                for field in table.schema:
                    value = row.get(field.name)
                    
                    # Convert BigQuery types to standard Python types
                    if value is not None:
                        if field.field_type in ['INTEGER', 'INT64']:
                            value = int(value)
                        elif field.field_type in ['FLOAT', 'FLOAT64', 'NUMERIC']:
                            value = float(value)
                        elif field.field_type == 'BOOLEAN':
                            value = bool(value)
                        elif field.field_type in ['TIMESTAMP', 'DATETIME', 'DATE']:
                            value = str(value)
                        elif field.field_type == 'RECORD':
                            value = dict(value) if value else None
                        elif field.field_type == 'REPEATED':
                            value = list(value) if value else []
                        else:
                            value = str(value) if value is not None else None
                    
                    row_dict[field.name] = value
                    
                    # Collect column samples
                    if field.name not in columns:
                        columns[field.name] = {
                            'name': field.name,
                            'type': field.field_type,
                            'mode': field.mode,
                            'samples': [],
                            'description': field.description
                        }
                    
                    if value is not None and len(columns[field.name]['samples']) < 100:
                        columns[field.name]['samples'].append(value)
                
                rows_list.append(row_dict)
            
            self.rows_processed += len(rows_list)
            
            # Analyze columns for discovery patterns
            logger.debug(f"        Analyzing {len(columns)} columns...")
            for col_name, col_info in columns.items():
                col_info['statistics'] = self._analyze_column(col_info['samples'])
                col_info['potential_type'] = self._infer_semantic_type(col_name, col_info['samples'])
                
                # Log interesting columns
                if col_info['potential_type'] in ['hostname', 'ip_address']:
                    logger.info(f"        🎯 Found {col_info['potential_type']} column: {col_name}")
            
            return {
                'name': table_id,
                'full_name': full_table_id,
                'rows': rows_list,
                'columns': columns,
                'row_count': len(rows_list),
                'total_rows': table.num_rows,
                'created': str(table.created) if table.created else None,
                'modified': str(table.modified) if table.modified else None,
                'size_bytes': table.num_bytes,
                'size_mb': round(table.num_bytes / 1024 / 1024, 2),
                'description': table.description
            }
            
        except exceptions.Forbidden as e:
            logger.error(f"        ❌ Permission denied reading table {full_table_id}")
            logger.error(f"           Need 'bigquery.tables.getData' permission")
            return None
            
        except Exception as e:
            logger.error(f"        ❌ Error scanning table {full_table_id}")
            logger.error(f"           Error type: {type(e).__name__}")
            logger.error(f"           Error: {e}")
            return None
    
    def _analyze_column(self, samples: List[Any]) -> Dict:
        """Analyze column samples for patterns"""
        if not samples:
            return {}
        
        stats = {
            'count': len(samples),
            'unique': len(set(str(s) for s in samples)),
            'null_count': sum(1 for s in samples if s is None),
            'unique_ratio': len(set(str(s) for s in samples)) / len(samples) if samples else 0
        }
        
        # Check for patterns
        non_null = [s for s in samples if s is not None]
        if non_null:
            # Check if numeric
            try:
                numeric_vals = [float(s) for s in non_null]
                stats['min'] = min(numeric_vals)
                stats['max'] = max(numeric_vals)
                stats['avg'] = sum(numeric_vals) / len(numeric_vals)
                stats['is_numeric'] = True
            except:
                stats['is_numeric'] = False
            
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
        
        # Check column name patterns (highest priority)
        if any(pattern in col_lower for pattern in ['hostname', 'host_name', 'fqdn', 'servername', 'server_name']):
            return 'hostname'
        elif any(pattern in col_lower for pattern in ['host', 'server', 'instance', 'node', 'machine', 'computer']) and 'name' in col_lower:
            return 'hostname'
        elif 'ip' in col_lower and any(pattern in col_lower for pattern in ['address', 'addr']):
            return 'ip_address'
        elif col_lower in ['ip', 'ipv4', 'ipv6', 'private_ip', 'public_ip', 'internal_ip', 'external_ip']:
            return 'ip_address'
        elif any(pattern in col_lower for pattern in ['environment', 'env', 'stage', 'tier']):
            return 'environment'
        elif any(pattern in col_lower for pattern in ['application', 'app', 'service']) and 'name' in col_lower:
            return 'application'
        elif any(pattern in col_lower for pattern in ['owner', 'team', 'department', 'contact']):
            return 'owner'
        elif any(pattern in col_lower for pattern in ['region', 'zone', 'datacenter', 'location', 'site']):
            return 'location'
        elif any(pattern in col_lower for pattern in ['project', 'gcp_project', 'aws_account']):
            return 'project'
        elif any(pattern in col_lower for pattern in ['cluster', 'pool', 'group']):
            return 'cluster'
        
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
            'rows_processed': self.rows_processed
        }