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
            logger.info(f"Default gcloud credentials")
            
            try:
                client = bigquery.Client(project=self.project_id)
                
                # Test the connection
                list(client.list_datasets(max_results=1))
                
                logger.info("Successfully authenticated using default credentials")
                return client
                
            except Exception as e:
                logger.error(f"All authentication methods failed: {e}")
                logger.error("Authentication methods tried:")
                logger.error("1. Service account keys gcp_prod_key.json")
                logger.error("2. GOOGLE_APPLICATION_CREDENTIALS environment variable")
                logger.error("3. Default gcloud credentials")
                logger.error("")
                logger.error("Fix this:")
                logger.error("Option 1 (Recommended): Place your service account key as")
                logger.error(f"'gcp_prod_key.json'")
                logger.error("Option 2: Set")
                logger.error("GOOGLE_APPLICATION_CREDENTIALS=/path/to/your/key.json")
                logger.error("Option 3: Run 'gcloud auth application-default login'")
                raise
        
        # 3. Finally check default gcloud location
        try:
            logger.info("Attempting default gcloud authentication")
            credentials_path = os.path.expanduser("~/.config/gcloud/application_default_credentials.json")
            
            if os.path.exists(credentials_path):
                credentials = service_account.Credentials.from_service_account_file(credentials_path)
                client = bigquery.Client(project=self.project_id, credentials=credentials)
            else:
                client = bigquery.Client(project=self.project_id)
            
            # Test the connection
            list(client.list_datasets(max_results=1))
            
            logger.info("Successfully authenticated using gcloud auth application-default login")
            return client
            
        except Exception as e:
            logger.error(f"All authentication methods failed: {e}")
            logger.error("Authentication methods tried:")
            logger.error("1. Service account keys gcp_prod_key.json")
            logger.error("2. GOOGLE_APPLICATION_CREDENTIALS environment variable")
            logger.error("3. Default gcloud credentials")
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
            logger.info(f"BigQuery connection test successful")
            return True
        except Exception as e:
            logger.error(f"BigQuery connection test failed: {e}")
            return False
    
    def get_project_info(self) -> Dict[str, Any]:
        try:
            client = self.get_client()
            project_id = client.project
            datasets = list(client.list_datasets(max_results=1))
            
            return {
                'project_id': project_id,
                'friendly_name': f"BigQuery Project: {project_id}",
                'description': f"Connected to BigQuery in project {project_id}"
            }
            
        except Exception as e:
            logger.error(f"Failed to get project info: {e}")
            return {}
    
    @contextmanager
    def get_connection(self):
        try:
            yield self.get_client()
        except Exception as e:
            logger.error(f"BigQuery operation failed: {e}")
            raise
        finally:
            pass  