# core/bigquery_scanner.py
"""
BigQuery Scanner - COMPLETELY REWRITTEN WITHOUT PANDAS/NUMPY - NO LIMITS
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
    """Scans BigQuery projects - NO PANDAS/NUMPY AT ALL - NO LIMITS"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.bq_config = config.get('bigquery', {})
        self.projects = self.bq_config.get('projects', [])
        self.sample_percent = self.bq_config.get('sample_percent', 100)  # Default to 100%
        self.max_rows_per_table = None  # NO LIMIT
        self.datasets_filter = set(self.bq_config.get('datasets_filter', []))
        self.tables_filter = set(self.bq_config.get('tables_filter', []))
        self.projects_scanned = []
        self.datasets_scanned = 0
        self.tables_scanned = 0
        self.rows_processed = 0
        self.tables_with_errors = []
        self.tables_with_hosts = []
        self.client_managers = {}
        
        if not BIGQUERY_AVAILABLE:
            raise ImportError("google-cloud-bigquery is required")
        
        if not self.projects:
            raise ValueError("No BigQuery projects configured")
        
        logger.info("=" * 80)
        logger.info("🚀 BigQuery Scanner initialized (NO PANDAS, NO LIMITS MODE)")
        logger.info(f"📋 Projects to scan: {', '.join(self.projects)}")
        logger.info("🔥 SCANNING ALL ROWS - NO LIMITS")
        logger.info("=" * 80)
    
    def _get_client(self, project_id: str) -> bigquery.Client:
        """Get or create BigQuery client"""
        if project_id not in self.client_managers:
            logger.info(f"🔑 Creating BigQuery client for project: {project_id}")
            self.client_managers[project_id] = BigQueryClientManager(project_id=project_id)
            if not self.client_managers[project_id].test_connection():
                raise ConnectionError(f"Failed to connect to BigQuery project: {project_id}")
            logger.info(f"✅ Connected to project: {project_id}")
        return self.client_managers[project_id].get_client()
    
    def _convert_value(self, value):
        """Convert ANY BigQuery value to a safe Python type"""
        if value is None:
            return None
        elif isinstance(value, (list, tuple)):
            # Convert lists/arrays to JSON string
            return json.dumps([self._convert_value(v) for v in value])
        elif isinstance(value, dict):
            # Convert dicts/structs to JSON string
            return json.dumps({k: self._convert_value(v) for k, v in value.items()})
        elif isinstance(value, bytes):
            return value.decode('utf-8', errors='ignore')
        elif isinstance(value, (datetime, )):
            return value.isoformat()
        else:
            # Everything else becomes a string
            return str(value)
    
    async def scan_all_projects(self) -> List[Dict]:
        """Scan all configured BigQuery projects"""
        all_data = []
        
        logger.info("\n" + "="*80)
        logger.info("📊 STARTING BIGQUERY SCAN (FULL SCAN - NO LIMITS)")
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
        
        logger.info("\n" + "="*80)
        logger.info("📊 SCAN SUMMARY")
        logger.info(f"✅ Projects scanned: {len(self.projects_scanned)}/{len(self.projects)}")
        logger.info(f"📁 Datasets scanned: {self.datasets_scanned}")
        logger.info(f"📋 Tables scanned: {self.tables_scanned}")
        logger.info(f"📝 Rows processed: {self.rows_processed:,}")
        logger.info(f"🏠 Tables with potential hosts: {len(self.tables_with_hosts)}")
        logger.info(f"⚠️  Tables with errors: {len(self.tables_with_errors)}")
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
                    logger.error(f"      ❌ Error scanning table {table_id}: {e}")
                    self.tables_with_errors.append(f"{project_id}.{dataset_id}.{table_id}")
                    
        except Exception as e:
            logger.error(f"    ❌ Error scanning dataset {dataset_id}: {e}")
        
        return dataset_data
    
    async def scan_table(self, client: bigquery.Client, project_id: str, dataset_id: str, table_id: str) -> Optional[Dict]:
        """
        COMPLETELY REWRITTEN: Scan table without pandas/numpy, NO ROW LIMITS
        """
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
            
            # Log column names
            column_names = [field.name for field in table.schema]
            logger.info(f"        📋 Columns: {', '.join(column_names[:10])}")
            if len(column_names) > 10:
                logger.info(f"           ... and {len(column_names)-10} more columns")
            
            # IMPORTANT: Build special query to handle arrays/structs
            select_parts = []
            complex_fields = []
            
            for field in table.schema:
                if field.mode == 'REPEATED' or field.field_type in ['RECORD', 'STRUCT', 'JSON', 'GEOGRAPHY', 'ARRAY']:
                    # Convert complex types to JSON strings IN THE QUERY
                    select_parts.append(f"TO_JSON_STRING({field.name}) AS {field.name}")
                    complex_fields.append(field.name)
                else:
                    select_parts.append(field.name)
            
            if complex_fields:
                logger.info(f"        ⚠️ Complex fields detected: {', '.join(complex_fields[:5])}")
                logger.info(f"           Converting to JSON strings for safe processing")
            
            # Build query - NO LIMIT IF WE WANT ALL DATA
            if self.sample_percent < 100:
                # Use sampling
                query = f"""
                SELECT {', '.join(select_parts)}
                FROM `{full_table_id}`
                TABLESAMPLE SYSTEM ({self.sample_percent} PERCENT)
                """
                logger.info(f"        🎯 Sampling {self.sample_percent}% of data")
            else:
                # GET EVERYTHING
                query = f"""
                SELECT {', '.join(select_parts)}
                FROM `{full_table_id}`
                """
                logger.info(f"        🔥 SCANNING ALL {table.num_rows:,} ROWS - NO LIMITS")
            
            logger.info(f"        ⏳ Executing query...")
            
            # Execute query
            query_job = client.query(query)
            
            # CRITICAL: Process results WITHOUT pandas
            rows = []
            columns_data = defaultdict(list)
            row_count = 0
            
            logger.info(f"        📥 Processing rows...")
            
            # Process in batches for memory efficiency
            batch_size = 10000
            batch = []
            
            for bq_row in query_job.result():
                row_dict = {}
                
                # Convert EVERY field to safe Python types
                for field in table.schema:
                    field_name = field.name
                    raw_value = bq_row.get(field_name)
                    
                    # Use our safe converter
                    safe_value = self._convert_value(raw_value)
                    
                    row_dict[field_name] = safe_value
                    
                    # Store samples for analysis (first 1000)
                    if len(columns_data[field_name]) < 1000:
                        columns_data[field_name].append(safe_value)
                
                batch.append(row_dict)
                row_count += 1
                
                # Process batches
                if len(batch) >= batch_size:
                    rows.extend(batch)
                    batch = []
                    
                    if row_count % 50000 == 0:
                        logger.info(f"        ... processed {row_count:,} rows")
            
            # Add remaining batch
            if batch:
                rows.extend(batch)
            
            logger.info(f"        ✅ Retrieved {row_count:,} rows")
            
            # Build columns metadata
            columns = {}
            for field in table.schema:
                col_name = field.name
                col_samples = columns_data.get(col_name, [])
                
                columns[col_name] = {
                    'name': col_name,
                    'type': field.field_type,
                    'mode': field.mode,
                    'samples': col_samples[:100],  # Keep first 100 samples
                    'description': field.description,
                    'statistics': self._analyze_column_safe(col_samples),
                    'potential_type': self._infer_semantic_type_safe(col_name, col_samples)
                }
                
                # Log interesting columns
                if columns[col_name]['potential_type'] in ['hostname', 'ip_address']:
                    logger.info(f"        🎯 Found {columns[col_name]['potential_type']} column: {col_name}")
            
            self.rows_processed += row_count
            
            return {
                'name': table_id,
                'full_name': full_table_id,
                'rows': rows,
                'columns': columns,
                'row_count': row_count,
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
            import traceback
            logger.error(f"           Traceback: {traceback.format_exc()}")
            raise
    
    def _check_for_hosts(self, table_data: Dict) -> bool:
        """Check if table likely contains host information"""
        columns = table_data.get('columns', {})
        
        for col_name, col_info in columns.items():
            if col_info.get('potential_type') in ['hostname', 'ip_address']:
                return True
        
        return False
    
    def _analyze_column_safe(self, samples: List[Any]) -> Dict:
        """Analyze column samples safely (all values are already strings or None)"""
        if not samples:
            return {}
        
        # Filter out None values
        non_null = []
        for s in samples:
            if s is not None:
                # If it's a JSON string, try to extract first element
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
                    non_null.append(str(s))
        
        if not non_null:
            return {'all_null': True}
        
        stats = {
            'count': len(samples),
            'unique': len(set(non_null)),
            'null_count': len(samples) - len(non_null),
            'unique_ratio': len(set(non_null)) / len(non_null) if non_null else 0
        }
        
        # Check for patterns
        if non_null:
            # Check for IP addresses
            ip_pattern = r'^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$'
            ip_matches = 0
            for s in non_null:
                try:
                    if re.match(ip_pattern, s):
                        ip_matches += 1
                except:
                    pass
            stats['ip_likelihood'] = ip_matches / len(non_null) if non_null else 0
            
            # Check for hostnames/FQDNs
            hostname_pattern = r'^[a-zA-Z0-9][a-zA-Z0-9\-\.]*[a-zA-Z0-9]$'
            hostname_matches = 0
            for s in non_null:
                try:
                    if re.match(hostname_pattern, s) and '.' in s:
                        hostname_matches += 1
                except:
                    pass
            stats['hostname_likelihood'] = hostname_matches / len(non_null) if non_null else 0
        
        return stats
    
    def _infer_semantic_type_safe(self, column_name: str, samples: List[Any]) -> str:
        """Infer semantic type (all values are already safe strings or None)"""
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
                    # Handle JSON strings
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
                        non_null.append(str(s))
            
            if non_null:
                # Check for IP pattern
                ip_matches = 0
                for s in non_null:
                    try:
                        if re.match(r'^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$', s):
                            ip_matches += 1
                    except:
                        pass
                
                if ip_matches > len(non_null) * 0.8:
                    return 'ip_address'
                
                # Check for hostname/FQDN pattern
                hostname_matches = 0
                for s in non_null:
                    try:
                        if re.match(r'^[a-zA-Z0-9][a-zA-Z0-9\-\.]*[a-zA-Z0-9]$', s) and '.' in s:
                            hostname_matches += 1
                    except:
                        pass
                
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