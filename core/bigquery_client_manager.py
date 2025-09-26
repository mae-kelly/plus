"""
BigQuery Client Manager - Handles authentication to BigQuery
"""

import os
import logging
from typing import Optional, Dict, Any
from google.cloud import bigquery
from google.oauth2 import service_account
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class BigQueryClientManager:
    
    def __init__(self, project_id: str = None):
        self.project_id = project_id
        self.client = None
        self.credentials = None
        
    def __create_client(self) -> bigquery.Client:
        credentials_paths = [
            os.path.join(os.path.dirname(__file__), "gcp_prod_key.json"),
            "gcp_prod_key.json",
        ]
        
        # 1. First check for gcp_prod_key.json in current directory
        for path in credentials_paths:
            if path and os.path.exists(path):
                try:
                    logger.info(f"Attempting service account authentication: {path}")
                    credentials = service_account.Credentials.from_service_account_file(path)
                    client = bigquery.Client(project=self.project_id, credentials=credentials)
                    
                    # Test the connection
                    list(client.list_datasets(max_results=1))
                    
                    logger.info(f"Successfully authenticated using service account: {path}")
                    return client
                    
                except Exception as e:
                    logger.error(f"Service account {path} failed: {e}")
                    continue
        
        # 2. Then check GOOGLE_APPLICATION_CREDENTIALS environment variable
        if 'GOOGLE_APPLICATION_CREDENTIALS' in os.environ:
            gcp_prod_key_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS', '')
            logger.info(f"Option 2: GOOGLE_APPLICATION_CREDENTIALS environment variable")
            logger.info(f"Path: {gcp_prod_key_path}")
            
            try:
                if os.path.exists(gcp_prod_key_path):
                    credentials = service_account.Credentials.from_service_account_file(gcp_prod_key_path)
                    client = bigquery.Client(project=self.project_id, credentials=credentials)
                else:
                    # Try default client with env var set
                    client = bigquery.Client(project=self.project_id)
                
                # Test the connection
                list(client.list_datasets(max_results=1))
                
                logger.info("Successfully authenticated using GOOGLE_APPLICATION_CREDENTIALS")
                return client
                
            except Exception as e:
                logger.debug(f"GOOGLE_APPLICATION_CREDENTIALS method failed: {e}")
        
        # 3. Finally check default gcloud location
        try:
            logger.info("Attempting default gcloud authentication")
            credentials_path = os.path.expanduser("~/.config/gcloud/application_default_credentials.json")
            
            if os.path.exists(credentials_path):
                logger.info(f"Found default credentials at: {credentials_path}")
                # Use application default credentials
                client = bigquery.Client(project=self.project_id)
            else:
                # Try to use default credentials anyway (might be in other locations)
                logger.info("Trying default credentials without specific file")
                client = bigquery.Client(project=self.project_id)
            
            # Test the connection
            list(client.list_datasets(max_results=1))
            
            logger.info("Successfully authenticated using gcloud auth application-default login")
            return client
            
        except Exception as e:
            logger.error(f"All authentication methods failed: {e}")
            logger.error("")
            logger.error("=" * 60)
            logger.error("AUTHENTICATION FAILED - Please use one of these methods:")
            logger.error("=" * 60)
            logger.error("")
            logger.error("Option 1 (Recommended): Service Account Key")
            logger.error("  1. Go to https://console.cloud.google.com")
            logger.error("  2. Navigate to IAM & Admin > Service Accounts")
            logger.error("  3. Create a service account or use existing")
            logger.error("  4. Create key (JSON format)")
            logger.error(f"  5. Save as 'gcp_prod_key.json' in current directory")
            logger.error("")
            logger.error("Option 2: Environment Variable")
            logger.error("  export GOOGLE_APPLICATION_CREDENTIALS=/path/to/your/key.json")
            logger.error("")
            logger.error("Option 3: Default Credentials")
            logger.error("  gcloud auth application-default login")
            logger.error("")
            logger.error("Required Permissions:")
            logger.error("  - BigQuery Data Viewer (roles/bigquery.dataViewer)")
            logger.error("  - BigQuery Job User (roles/bigquery.jobUser)")
            logger.error("=" * 60)
            raise
    
    def get_client(self, project_id: str = None) -> bigquery.Client:
        if project_id:
            self.project_id = project_id
            
        if self.client is None:
            self.client = self.__create_client()
            
        return self.client
    
    def test_connection(self) -> bool:
        try:
            client = self.get_client()
            datasets = list(client.list_datasets(max_results=1))
            logger.info(f"BigQuery connection test successful for project: {self.project_id}")
            return True
        except Exception as e:
            logger.error(f"BigQuery connection test failed: {e}")
            return False
    
    def get_project_info(self) -> Dict[str, Any]:
        try:
            client = self.get_client()
            project_id = client.project
            
            # Try to get more project info
            dataset_count = 0
            try:
                datasets = list(client.list_datasets(max_results=100))
                dataset_count = len(datasets)
            except:
                pass
            
            return {
                'project_id': project_id,
                'friendly_name': f"BigQuery Project: {project_id}",
                'description': f"Connected to BigQuery in project {project_id}",
                'dataset_count': dataset_count
            }
            
        except Exception as e:
            logger.error(f"Failed to get project info: {e}")
            return {}
    
    @contextmanager
    def get_connection(self):
        """Context manager for BigQuery operations"""
        try:
            yield self.get_client()
        except Exception as e:
            logger.error(f"BigQuery operation failed: {e}")
            raise
        finally:
            pass  # BigQuery client doesn't need explicit closing