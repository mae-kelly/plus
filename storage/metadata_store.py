import json
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

class MetadataStore:
    """
    Simplified metadata store that only uses local file storage.
    No Elasticsearch or Neo4j dependencies.
    """
    def __init__(self, config: Dict):
        self.config = config
        self.metadata_file = 'metadata.json'
        self.hosts_file = 'hosts.json'
        
        # No external clients
        self.es_client = None
        self.neo4j_driver = None
        
        logger.info("MetadataStore initialized (local file storage only)")
    
    async def persist_metadata(self, column_metadata: Dict):
        """Save metadata to local JSON file"""
        metadata_doc = {
            'columns': [],
            'statistics': {}
        }
        
        for column_name, metadata in column_metadata.items():
            # Convert sets to lists for JSON serialization
            tables = metadata.get('tables', set())
            if isinstance(tables, set):
                tables = list(tables)
            
            column_doc = {
                'name': column_name,
                'type': metadata.get('type', 'unknown'),
                'confidence': metadata.get('confidence', 0),
                'frequency': metadata.get('frequency', 0),
                'tables': tables
            }
            metadata_doc['columns'].append(column_doc)
        
        # Always save to local file
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata_doc, f, indent=2)
        
        logger.info(f"Metadata saved to {self.metadata_file}")
    
    async def index_hosts(self, hosts: List[Dict]):
        """Save hosts to local JSON file"""
        # Save to local file
        with open(self.hosts_file, 'w') as f:
            json.dump(hosts, f, indent=2)
        
        logger.info(f"Indexed {len(hosts)} hosts to {self.hosts_file}")
    
    async def create_graph_relationships(self, relationships: List[Dict]):
        """Save relationships to local file"""
        with open('relationships.json', 'w') as f:
            json.dump(relationships, f, indent=2)
        
        logger.info(f"Created {len(relationships)} relationships in relationships.json")
    
    def search_hosts(self, query: str, size: int = 100) -> List[Dict]:
        """Simple search in local hosts file"""
        try:
            with open(self.hosts_file, 'r') as f:
                hosts = json.load(f)
            
            results = []
            query_lower = query.lower()
            
            for host in hosts:
                if query_lower in str(host).lower():
                    results.append(host)
                    if len(results) >= size:
                        break
            
            return results
            
        except FileNotFoundError:
            logger.warning(f"Hosts file {self.hosts_file} not found")
            return []
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def get_host_graph(self, hostname: str, depth: int = 2) -> Dict:
        """Return empty graph (no Neo4j)"""
        return {
            'nodes': [hostname],
            'edges': []
        }
    
    def close(self):
        """No external connections to close"""
        pass