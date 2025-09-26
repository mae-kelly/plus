"""
BigQuery Scanner - Connects to and scans real BigQuery projects
"""

import asyncio
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
import re

logger = logging.getLogger(__name__)

try:
    from google.cloud import bigquery
    from google.oauth2 import service_account
    from google.api_core import exceptions
    BIGQUERY_AVAILABLE = True
except ImportError:
    BIGQUERY_AVAILABLE = False
    logger.error("❌ Google Cloud BigQuery not installed!")
    logger.error("Run: pip install google-cloud-bigquery")

class BigQueryScanner:
    """Scans BigQuery projects for infrastructure data"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.bq_config = config.get('bigquery', {})
        
        # BigQuery settings
        self.projects = self.bq_config.get('projects', [])
        self.credentials_path = self.bq_config.get('credentials_path', 'gcp_credentials.json')
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
        
        # Clients cache
        self.clients = {}
        
        # Validate setup
        if not BIGQUERY_AVAILABLE:
            raise ImportError("google-cloud-bigquery is required. Install it with: pip install google-cloud-bigquery")
        
        if not self.projects:
            raise ValueError("No BigQuery projects configured. Add project IDs to config.json")
    
    def _get_client(self, project_id: str) -> bigquery.Client:
        """Get or create BigQuery client"""
        if project_id not in self.clients:
            try:
                from pathlib import Path
                
                # Try credentials file
                if Path(self.credentials_path).exists():
                    logger.info(f"Using credentials from {self.credentials_path}")
                    credentials = service_account.Credentials.from_service_account_file(
                        self.credentials_path
                    )
                    self.clients[project_id] = bigquery.Client(
                        project=project_id,
                        credentials=credentials
                    )
                else:
                    # Use default credentials
                    logger.info(f"Using default credentials for project {project_id}")
                    self.clients[project_id] = bigquery.Client(project=project_id)
                    
            except Exception as e:
                logger.error(f"Failed to create BigQuery client for {project_id}: {e}")
                raise
        
        return self.clients[project_id]
    
    async def scan_all_projects(self) -> List[Dict]:
        """Scan all configured BigQuery projects"""
        all_data = []
        
        for project_id in self.projects:
            logger.info(f"📊 Scanning BigQuery project: {project_id}")
            
            try:
                project_data = await self.scan_project(project_id)
                all_data.extend(project_data)
                self.projects_scanned.append(project_id)
                
            except exceptions.Forbidden as e:
                logger.error(f"❌ Permission denied for project {project_id}: {e}")
                logger.error("Make sure the service account has BigQuery Data Viewer role")
            except Exception as e:
                logger.error(f"❌ Error scanning project {project_id}: {e}")
        
        return all_data
    
    async def scan_project(self, project_id: str) -> List[Dict]:
        """Scan a single BigQuery project"""
        client = self._get_client(project_id)
        project_data = []
        
        # List datasets
        try:
            datasets = list(client.list_datasets(timeout=30))
            logger.info(f"Found {len(datasets)} datasets in {project_id}")
            
            for dataset_ref in datasets:
                dataset_id = dataset_ref.dataset_id
                
                # Apply filter
                if self.datasets_filter and dataset_id not in self.datasets_filter:
                    continue
                
                logger.info(f"  📁 Scanning dataset: {dataset_id}")
                
                dataset_data = await self.scan_dataset(client, project_id, dataset_id)
                if dataset_data:
                    project_data.extend(dataset_data)
                    self.datasets_scanned += 1
                    
        except Exception as e:
            logger.error(f"Error listing datasets in {project_id}: {e}")
        
        return project_data
    
    async def scan_dataset(self, client: bigquery.Client, project_id: str, dataset_id: str) -> List[Dict]:
        """Scan all tables in a dataset"""
        dataset_data = []
        
        try:
            # List tables
            tables = list(client.list_tables(f"{project_id}.{dataset_id}", timeout=30))
            logger.info(f"    Found {len(tables)} tables in {dataset_id}")
            
            for table_ref in tables:
                table_id = table_ref.table_id
                
                # Apply filter
                if self.tables_filter and table_id not in self.tables_filter:
                    continue
                
                logger.debug(f"      📋 Scanning table: {table_id}")
                
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
            logger.error(f"Error scanning dataset {dataset_id}: {e}")
        
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
            
            # Determine how many rows to sample
            sample_size = min(
                int(table.num_rows * self.sample_percent / 100),
                self.max_rows_per_table
            )
            
            logger.debug(f"        Sampling {sample_size:,} of {table.num_rows:,} rows")
            
            # Build query
            if sample_size < table.num_rows:
                # Use TABLESAMPLE for efficient sampling
                query = f"""
                SELECT *
                FROM `{full_table_id}`
                TABLESAMPLE SYSTEM ({self.sample_percent} PERCENT)
                LIMIT {sample_size}
                """
            else:
                # Get all rows for small tables
                query = f"""
                SELECT *
                FROM `{full_table_id}`
                """
            
            # Execute query
            query_job = client.query(query)
            rows = list(query_job.result(timeout=60))
            
            if not rows:
                return None
            
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
                        elif field.field_type in ['TIMESTAMP', 'DATETIME']:
                            value = str(value)
                        else:
                            value = str(value)
                    
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
            for col_name, col_info in columns.items():
                col_info['statistics'] = self._analyze_column(col_info['samples'])
                col_info['potential_type'] = self._infer_semantic_type(col_name, col_info['samples'])
            
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
                'description': table.description
            }
            
        except Exception as e:
            logger.error(f"Error scanning table {full_table_id}: {e}")
            return None
    
    def _analyze_column(self, samples: List[Any]) -> Dict:
        """Analyze column samples for patterns"""
        if not samples:
            return {}
        
        stats = {
            'count': len(samples),
            'unique': len(set(str(s) for s in samples)),
            'null_count': sum(1 for s in samples if s is None),
            'unique_ratio': len(set(str(s) for s in samples)) / len(samples)
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
            stats['ip_likelihood'] = ip_matches / len(non_null)
            
            # Check for hostnames
            hostname_pattern = r'^[a-zA-Z0-9][a-zA-Z0-9\-\.]*[a-zA-Z0-9]$'
            hostname_matches = sum(1 for s in non_null if re.match(hostname_pattern, str(s)))
            stats['hostname_likelihood'] = hostname_matches / len(non_null)
        
        return stats
    
    def _infer_semantic_type(self, column_name: str, samples: List[Any]) -> str:
        """Infer the semantic type of a column"""
        col_lower = column_name.lower()
        
        # Check column name patterns
        if any(pattern in col_lower for pattern in ['host', 'server', 'instance', 'node', 'machine']):
            return 'hostname'
        elif 'ip' in col_lower or 'address' in col_lower:
            return 'ip_address'
        elif 'env' in col_lower:
            return 'environment'
        elif 'app' in col_lower or 'service' in col_lower:
            return 'application'
        elif 'owner' in col_lower or 'team' in col_lower:
            return 'owner'
        elif 'region' in col_lower or 'zone' in col_lower or 'location' in col_lower:
            return 'location'
        elif 'project' in col_lower:
            return 'project'
        elif 'cluster' in col_lower:
            return 'cluster'
        
        # Check sample patterns
        if samples:
            non_null = [s for s in samples[:100] if s is not None]
            if non_null:
                # Check for IP pattern
                ip_matches = sum(1 for s in non_null 
                                if re.match(r'^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$', str(s)))
                if ip_matches > len(non_null) * 0.8:
                    return 'ip_address'
                
                # Check for hostname pattern
                hostname_matches = sum(1 for s in non_null 
                                     if re.match(r'^[a-zA-Z0-9][a-zA-Z0-9\-\.]*[a-zA-Z0-9]$', str(s)))
                if hostname_matches > len(non_null) * 0.6:
                    return 'hostname'
        
        return 'unknown'