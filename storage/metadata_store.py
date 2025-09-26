import json
from typing import Dict, List, Any
from elasticsearch import Elasticsearch
from neo4j import GraphDatabase
import logging

logger = logging.getLogger(__name__)

class MetadataStore:
    def __init__(self, config: Dict):
        self.config = config
        
        self.es_client = None
        if config['storage'].get('elasticsearch_host'):
            # Fix: Ensure proper URL format for Elasticsearch
            es_host = config['storage']['elasticsearch_host']
            
            # Add http:// if no scheme is present
            if not es_host.startswith(('http://', 'https://')):
                es_host = f'http://{es_host}'
            
            try:
                self.es_client = Elasticsearch([es_host])
                self._init_elasticsearch()
            except Exception as e:
                logger.warning(f"Elasticsearch connection failed: {e}. Continuing without Elasticsearch.")
                self.es_client = None
        
        self.neo4j_driver = None
        if config['storage'].get('neo4j_uri'):
            try:
                self.neo4j_driver = GraphDatabase.driver(
                    config['storage']['neo4j_uri'],
                    auth=(config['storage'].get('neo4j_user', 'neo4j'),
                          config['storage'].get('neo4j_password', 'password'))
                )
            except Exception as e:
                logger.warning(f"Neo4j connection failed: {e}. Continuing without Neo4j.")
                self.neo4j_driver = None
    
    def _init_elasticsearch(self):
        try:
            index_config = {
                'settings': {
                    'number_of_shards': 3,
                    'number_of_replicas': 1,
                    'analysis': {
                        'analyzer': {
                            'hostname_analyzer': {
                                'tokenizer': 'hostname_tokenizer',
                                'filter': ['lowercase', 'stop']
                            }
                        },
                        'tokenizer': {
                            'hostname_tokenizer': {
                                'type': 'pattern',
                                'pattern': '[._-]'
                            }
                        }
                    }
                },
                'mappings': {
                    'properties': {
                        'hostname': {
                            'type': 'text',
                            'analyzer': 'hostname_analyzer',
                            'fields': {
                                'keyword': {'type': 'keyword'}
                            }
                        },
                        'raw_forms': {'type': 'text'},
                        'confidence': {'type': 'float'},
                        'quality_score': {'type': 'float'},
                        'discovered_at': {'type': 'date'},
                        'attributes': {'type': 'object', 'enabled': False},
                        'environment': {'type': 'keyword'},
                        'datacenter': {'type': 'keyword'},
                        'os_type': {'type': 'keyword'}
                    }
                }
            }
            
            if not self.es_client.indices.exists(index='cmdb_hosts'):
                self.es_client.indices.create(index='cmdb_hosts', body=index_config)
                logger.info("Elasticsearch index created")
        except Exception as e:
            logger.error(f"Failed to initialize Elasticsearch index: {e}")
    
    async def persist_metadata(self, column_metadata: Dict):
        metadata_doc = {
            'columns': [],
            'statistics': {}
        }
        
        for column_name, metadata in column_metadata.items():
            column_doc = {
                'name': column_name,
                'type': metadata.get('type', 'unknown'),
                'confidence': metadata.get('confidence', 0),
                'frequency': metadata.get('frequency', 0),
                'tables': list(metadata.get('tables', set()))
            }
            metadata_doc['columns'].append(column_doc)
        
        if self.es_client:
            try:
                self.es_client.index(
                    index='cmdb_metadata',
                    body=metadata_doc,
                    id='column_metadata'
                )
                logger.info("Metadata persisted to Elasticsearch")
            except Exception as e:
                logger.error(f"Failed to persist metadata to Elasticsearch: {e}")
        
        # Always save to file as backup
        with open('metadata.json', 'w') as f:
            json.dump(metadata_doc, f, indent=2)
    
    async def index_hosts(self, hosts: List[Dict]):
        if not self.es_client:
            return
        
        for host in hosts:
            try:
                doc = {
                    'hostname': host['hostname'],
                    'raw_forms': host.get('raw_forms', []),
                    'confidence': host.get('confidence', 0),
                    'quality_score': host.get('quality_score', 0),
                    'discovered_at': host.get('discovered_at'),
                    'attributes': host.get('attributes', {}),
                    'environment': host.get('environment'),
                    'datacenter': host.get('datacenter'),
                    'os_type': host.get('os_type')
                }
                
                self.es_client.index(
                    index='cmdb_hosts',
                    body=doc,
                    id=host['hostname']
                )
                
            except Exception as e:
                logger.error(f"Failed to index host: {e}")
        
        logger.info(f"Indexed {len(hosts)} hosts to Elasticsearch")
    
    async def create_graph_relationships(self, relationships: List[Dict]):
        if not self.neo4j_driver:
            return
        
        try:
            with self.neo4j_driver.session() as session:
                for rel in relationships:
                    try:
                        session.run("""
                            MERGE (a:Host {name: $source})
                            MERGE (b:Host {name: $target})
                            MERGE (a)-[r:RELATED {
                                type: $rel_type,
                                confidence: $confidence
                            }]->(b)
                        """, source=rel['source'],
                             target=rel['target'],
                             rel_type=rel['type'],
                             confidence=rel.get('confidence', 0.5))
                        
                    except Exception as e:
                        logger.error(f"Failed to create graph relationship: {e}")
            
            logger.info(f"Created {len(relationships)} graph relationships")
        except Exception as e:
            logger.error(f"Failed to create graph relationships: {e}")
    
    def search_hosts(self, query: str, size: int = 100) -> List[Dict]:
        if not self.es_client:
            return []
        
        search_body = {
            'query': {
                'multi_match': {
                    'query': query,
                    'fields': ['hostname', 'raw_forms', 'attributes.*']
                }
            },
            'size': size
        }
        
        try:
            response = self.es_client.search(index='cmdb_hosts', body=search_body)
            
            results = []
            for hit in response['hits']['hits']:
                results.append(hit['_source'])
            
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def get_host_graph(self, hostname: str, depth: int = 2) -> Dict:
        if not self.neo4j_driver:
            return {}
        
        try:
            with self.neo4j_driver.session() as session:
                result = session.run("""
                    MATCH path = (h:Host {name: $hostname})-[*1..$depth]-(connected)
                    RETURN path
                """, hostname=hostname, depth=depth)
                
                nodes = set()
                edges = []
                
                for record in result:
                    path = record['path']
                    
                    for node in path.nodes:
                        nodes.add(node['name'])
                    
                    for rel in path.relationships:
                        edges.append({
                            'source': rel.start_node['name'],
                            'target': rel.end_node['name'],
                            'type': rel.get('type', 'related')
                        })
                
                return {
                    'nodes': list(nodes),
                    'edges': edges
                }
        except Exception as e:
            logger.error(f"Failed to get host graph: {e}")
            return {}
    
    def close(self):
        if self.neo4j_driver:
            try:
                self.neo4j_driver.close()
            except:
                pass