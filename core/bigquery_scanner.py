import asyncio
from typing import Dict, List, AsyncGenerator, Any
import logging

logger = logging.getLogger(__name__)

# Try to import BigQuery, but handle if not available
try:
    from google.cloud import bigquery
    from google.oauth2 import service_account
    BIGQUERY_AVAILABLE = True
except ImportError:
    BIGQUERY_AVAILABLE = False
    logger.warning("Google Cloud BigQuery not available")

class BigQueryScanner:
    def __init__(self, config: Dict):
        self.config = config
        self.clients = {}
        self.batch_size = config.get('batch_size', 128)
        self.max_rows = config['sampling']['max_rows_per_table']
        
        # Check if we can actually use BigQuery
        self.enabled = self._check_availability()
        
    def _check_availability(self) -> bool:
        """Check if BigQuery can be used"""
        if not BIGQUERY_AVAILABLE:
            logger.warning("BigQuery libraries not installed")
            return False
        
        # Check for valid project IDs
        projects = self.config.get('projects', [])
        if not projects or projects == ['your-project-id-1', 'your-project-id-2'] or projects == ['test-project']:
            logger.warning("No valid BigQuery project IDs configured")
            return False
        
        # Check for credentials
        try:
            import os
            from pathlib import Path
            
            # Check various credential sources
            if Path('gcp_prod_key.json').exists():
                logger.info("Found GCP credentials at gcp_prod_key.json")
                return True
            elif os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'):
                logger.info("Found GOOGLE_APPLICATION_CREDENTIALS environment variable")
                return True
            else:
                # Try default credentials
                from google.auth import default
                credentials, project = default()
                logger.info("Using default GCP credentials")
                return True
                
        except Exception as e:
            logger.warning(f"No valid GCP credentials found: {e}")
            return False
        
    def _get_client(self, project_id: str):
        """Get BigQuery client for project"""
        if not self.enabled:
            return None
            
        if project_id not in self.clients:
            try:
                # Try to load credentials
                from pathlib import Path
                if Path('gcp_prod_key.json').exists():
                    credentials = service_account.Credentials.from_service_account_file(
                        'gcp_prod_key.json'
                    )
                    self.clients[project_id] = bigquery.Client(
                        project=project_id,
                        credentials=credentials
                    )
                else:
                    # Use default credentials
                    self.clients[project_id] = bigquery.Client(project=project_id)
                    
            except Exception as e:
                logger.error(f"Failed to create BigQuery client for {project_id}: {e}")
                return None
                
        return self.clients[project_id]
    
    async def scan_projects(self, project_ids: List[str]) -> AsyncGenerator:
        """Scan BigQuery projects or return empty if not available"""
        if not self.enabled:
            logger.warning("BigQuery scanning is disabled - no valid configuration")
            # Return empty generator
            return
            yield  # Make this a generator
            
        for project_id in project_ids:
            logger.info(f"Scanning BigQuery project: {project_id}")
            try:
                async for batch in self._scan_project(project_id):
                    yield batch
            except Exception as e:
                logger.error(f"Error scanning project {project_id}: {e}")
                # Continue with next project instead of failing
                continue
    
    async def _scan_project(self, project_id: str) -> AsyncGenerator:
        """Scan a single project"""
        client = self._get_client(project_id)
        if not client:
            logger.warning(f"No client available for project {project_id}")
            return
        
        try:
            # List datasets with timeout
            datasets = []
            try:
                # Add timeout to prevent hanging
                datasets = list(client.list_datasets(timeout=30))
            except Exception as e:
                logger.error(f"Failed to list datasets in {project_id}: {e}")
                return
            
            if not datasets:
                logger.info(f"No datasets found in project {project_id}")
                return
            
            logger.info(f"Found {len(datasets)} datasets in {project_id}")
            
            for dataset_ref in datasets:
                logger.info(f"Scanning dataset: {dataset_ref.dataset_id}")
                async for batch in self._scan_dataset(client, project_id, dataset_ref.dataset_id):
                    yield batch
                    
        except Exception as e:
            logger.error(f"Error scanning project {project_id}: {e}")
    
    async def _scan_dataset(self, client, project_id: str, dataset_id: str) -> AsyncGenerator:
        """Scan a single dataset"""
        try:
            tables = []
            try:
                # Add timeout to prevent hanging
                tables = list(client.list_tables(f"{project_id}.{dataset_id}", timeout=30))
            except Exception as e:
                logger.error(f"Failed to list tables in {dataset_id}: {e}")
                return
            
            if not tables:
                logger.info(f"No tables found in dataset {dataset_id}")
                return
                
            logger.info(f"Found {len(tables)} tables in {dataset_id}")
            
            batch = []
            for table_ref in tables:
                logger.debug(f"Scanning table: {table_ref.table_id}")
                
                # Add async sleep to prevent blocking
                await asyncio.sleep(0)
                
                table_data = await self._scan_table(client, project_id, dataset_id, table_ref.table_id)
                
                if table_data:
                    batch.append(table_data)
                    
                    if len(batch) >= self.batch_size:
                        logger.info(f"Yielding batch of {len(batch)} tables")
                        yield batch
                        batch = []
            
            if batch:
                logger.info(f"Yielding final batch of {len(batch)} tables")
                yield batch
                
        except Exception as e:
            logger.error(f"Error scanning dataset {dataset_id}: {e}")
    
    async def _scan_table(self, client, project_id: str, dataset_id: str, table_id: str):
        """Scan a single table"""
        full_table_id = f"{project_id}.{dataset_id}.{table_id}"
        
        try:
            # Get table metadata
            table = client.get_table(full_table_id, timeout=30)
            
            if table.num_rows == 0:
                logger.debug(f"Table {table_id} is empty")
                return None
            
            # Initialize columns dictionary
            columns = {field.name: [] for field in table.schema}
            
            # Sample rows from table
            sample_size = min(self.max_rows, table.num_rows)
            query = f"""
            SELECT *
            FROM `{full_table_id}`
            LIMIT {sample_size}
            """
            
            logger.debug(f"Querying {sample_size} rows from {table_id}")
            
            # Run query with timeout
            query_job = client.query(query, timeout=30)
            rows = []
            
            # Fetch results
            try:
                for row in query_job.result(timeout=30):
                    row_dict = dict(row)
                    rows.append(row_dict)
                    
                    for column, value in row_dict.items():
                        if column in columns:
                            columns[column].append(value)
            except Exception as e:
                logger.error(f"Failed to fetch rows from {table_id}: {e}")
                return None
            
            logger.debug(f"Retrieved {len(rows)} rows from {table_id}")
            
            return {
                'table_name': full_table_id,
                'rows': rows,
                'columns': columns,
                'schema': [{'name': f.name, 'type': f.field_type} for f in table.schema]
            }
            
        except Exception as e:
            logger.debug(f"Error scanning table {table_id}: {e}")
            return None


class EmptyScanner:
    """Empty scanner for when BigQuery is not available"""
    def __init__(self, config: Dict):
        self.config = config
        logger.info("Using EmptyScanner - no data will be scanned")
    
    async def scan_projects(self, project_ids: List[str]) -> AsyncGenerator:
        """Return empty results"""
        logger.warning("No data source configured - returning empty results")
        # Return empty generator
        return
        yield  # Make this a generator