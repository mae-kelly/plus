import asyncio
from typing import Dict, List, AsyncGenerator, Any
from google.cloud import bigquery
from google.oauth2 import service_account
import logging

logger = logging.getLogger(__name__)

class BigQueryScanner:
    def __init__(self, config: Dict):
        self.config = config
        self.clients = {}
        self.batch_size = config.get('batch_size', 128)
        self.max_rows = config['sampling']['max_rows_per_table']
        
    def _get_client(self, project_id: str):
        if project_id not in self.clients:
            credentials = service_account.Credentials.from_service_account_file(
                'gcp_prod_key.json'
            )
            self.clients[project_id] = bigquery.Client(
                project=project_id,
                credentials=credentials
            )
        return self.clients[project_id]
    
    async def scan_projects(self, project_ids: List[str]) -> AsyncGenerator:
        for project_id in project_ids:
            async for batch in self._scan_project(project_id):
                yield batch
    
    async def _scan_project(self, project_id: str) -> AsyncGenerator:
        client = self._get_client(project_id)
        
        try:
            datasets = list(client.list_datasets())
            
            for dataset_ref in datasets:
                async for batch in self._scan_dataset(client, project_id, dataset_ref.dataset_id):
                    yield batch
                    
        except Exception as e:
            logger.error(f"Error scanning project {project_id}: {e}")
    
    async def _scan_dataset(self, client, project_id: str, dataset_id: str) -> AsyncGenerator:
        try:
            tables = list(client.list_tables(f"{project_id}.{dataset_id}"))
            
            batch = []
            for table_ref in tables:
                table_data = await self._scan_table(client, project_id, dataset_id, table_ref.table_id)
                
                if table_data:
                    batch.append(table_data)
                    
                    if len(batch) >= self.batch_size:
                        yield batch
                        batch = []
            
            if batch:
                yield batch
                
        except Exception as e:
            logger.error(f"Error scanning dataset {dataset_id}: {e}")
    
    async def _scan_table(self, client, project_id: str, dataset_id: str, table_id: str):
        full_table_id = f"{project_id}.{dataset_id}.{table_id}"
        
        try:
            table = client.get_table(full_table_id)
            
            if table.num_rows == 0:
                return None
            
            columns = {field.name: [] for field in table.schema}
            
            query = f"""
            SELECT *
            FROM `{full_table_id}`
            LIMIT {min(self.max_rows, table.num_rows)}
            """
            
            query_job = client.query(query)
            rows = []
            
            for row in query_job:
                row_dict = dict(row)
                rows.append(row_dict)
                
                for column, value in row_dict.items():
                    if column in columns:
                        columns[column].append(value)
            
            return {
                'table_name': full_table_id,
                'rows': rows,
                'columns': columns,
                'schema': [{'name': f.name, 'type': f.field_type} for f in table.schema]
            }
            
        except Exception as e:
            logger.debug(f"Error scanning table {table_id}: {e}")
            return None