"""
Discovery Orchestrator - Manages the entire discovery pipeline
"""

import asyncio
import json
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from collections import defaultdict
import logging

from core.scanner import DataScanner
from core.classifier import IntelligentClassifier
from core.relationship_mapper import RelationshipMapper
from core.cmdb_builder import CMDBBuilder
from core.quality_analyzer import QualityAnalyzer

logger = logging.getLogger(__name__)

class DiscoveryOrchestrator:
    """Orchestrates the entire discovery process"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.start_time = datetime.now()
        
        # Initialize components
        self.scanner = DataScanner(config)
        self.classifier = IntelligentClassifier()
        self.relationship_mapper = RelationshipMapper()
        self.cmdb_builder = CMDBBuilder(config.get('output_database', 'cmdb.db'))
        self.quality_analyzer = QualityAnalyzer()
        
        # State tracking
        self.state = {
            'phase': 'initialized',
            'discovered_hosts': {},
            'classified_entities': {},
            'relationships': [],
            'statistics': defaultdict(int)
        }
        
        self.checkpoint_file = Path('checkpoints/discovery_state.pkl')
        self.checkpoint_file.parent.mkdir(exist_ok=True)
    
    async def scan_data_sources(self) -> List[Dict]:
        """Scan all configured data sources"""
        self.state['phase'] = 'scanning'
        discovered_data = []
        
        # Scan different source types
        sources = self.config.get('data_sources', {})
        
        # CSV files
        if sources.get('csv_files'):
            csv_data = await self.scanner.scan_csv_files(sources['csv_files'])
            discovered_data.extend(csv_data)
            logger.info(f"Scanned {len(csv_data)} CSV files")
        
        # JSON files
        if sources.get('json_files'):
            json_data = await self.scanner.scan_json_files(sources['json_files'])
            discovered_data.extend(json_data)
            logger.info(f"Scanned {len(json_data)} JSON files")
        
        # Databases
        if sources.get('databases'):
            db_data = await self.scanner.scan_databases(sources['databases'])
            discovered_data.extend(db_data)
            logger.info(f"Scanned {len(db_data)} databases")
        
        # API endpoints
        if sources.get('apis'):
            api_data = await self.scanner.scan_apis(sources['apis'])
            discovered_data.extend(api_data)
            logger.info(f"Scanned {len(api_data)} APIs")
        
        # If no sources configured, scan default locations
        if not discovered_data:
            logger.info("No data sources configured, scanning default locations...")
            discovered_data = await self.scanner.scan_default_locations()
        
        self.state['statistics']['data_sources'] = len(discovered_data)
        return discovered_data
    
    async def extract_hosts(self, data_sources: List[Dict]) -> Dict[str, Dict]:
        """Extract potential hosts from data sources"""
        self.state['phase'] = 'extracting'
        hosts = {}
        
        for source in data_sources:
            source_type = source.get('type', 'unknown')
            source_name = source.get('name', 'unnamed')
            
            logger.debug(f"Processing source: {source_name} ({source_type})")
            
            # Process tables/collections
            for table in source.get('tables', []):
                table_name = table.get('name', 'unknown')
                rows = table.get('rows', [])
                columns = table.get('columns', {})
                
                self.state['statistics']['tables_processed'] += 1
                self.state['statistics']['rows_scanned'] += len(rows)
                
                # Analyze each row for potential hosts
                for row in rows:
                    potential_hosts = self._extract_hosts_from_row(row, columns, table_name)
                    
                    for host_info in potential_hosts:
                        hostname = host_info['hostname']
                        
                        if hostname not in hosts:
                            hosts[hostname] = {
                                'hostname': hostname,
                                'sources': [],
                                'attributes': defaultdict(list),
                                'confidence': 0.0,
                                'first_seen': datetime.now().isoformat(),
                                'last_seen': datetime.now().isoformat()
                            }
                        
                        # Add source reference
                        hosts[hostname]['sources'].append({
                            'source': source_name,
                            'table': table_name,
                            'column': host_info['column'],
                            'confidence': host_info['confidence']
                        })
                        
                        # Add attributes from row
                        for col, value in row.items():
                            if col != host_info['column'] and value:
                                hosts[hostname]['attributes'][col].append(value)
                        
                        # Update confidence
                        hosts[hostname]['confidence'] = max(
                            hosts[hostname]['confidence'],
                            host_info['confidence']
                        )
        
        self.state['discovered_hosts'] = hosts
        self.state['statistics']['hosts_discovered'] = len(hosts)
        
        logger.info(f"Extracted {len(hosts)} unique hosts")
        return hosts
    
    def _extract_hosts_from_row(self, row: Dict, columns: Dict, table_name: str) -> List[Dict]:
        """Extract potential hosts from a data row"""
        potential_hosts = []
        
        for column, value in row.items():
            if not value:
                continue
            
            # Check if this could be a hostname
            confidence = self._calculate_hostname_confidence(column, value, columns.get(column, {}))
            
            if confidence > self.config.get('hostname_confidence_threshold', 0.6):
                potential_hosts.append({
                    'hostname': self._normalize_hostname(value),
                    'column': column,
                    'confidence': confidence,
                    'raw_value': value
                })
        
        return potential_hosts
    
    def _calculate_hostname_confidence(self, column_name: str, value: Any, column_metadata: Dict) -> float:
        """Calculate confidence that a value is a hostname"""
        confidence = 0.0
        value_str = str(value).lower().strip()
        column_lower = column_name.lower()
        
        # Column name indicators
        hostname_keywords = ['host', 'server', 'node', 'machine', 'computer', 'instance', 'device']
        for keyword in hostname_keywords:
            if keyword in column_lower:
                confidence += 0.3
                break
        
        # Value pattern analysis
        import re
        
        # FQDN pattern
        if re.match(r'^[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?(\.[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?)*$', value_str):
            confidence += 0.4
            if value_str.count('.') >= 1:  # Has domain
                confidence += 0.2
        
        # Simple hostname pattern
        elif re.match(r'^[a-z0-9][a-z0-9-]*[a-z0-9]$', value_str):
            confidence += 0.3
        
        # IP address pattern
        elif re.match(r'^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$', value_str):
            octets = value_str.split('.')
            if all(0 <= int(octet) <= 255 for octet in octets):
                confidence += 0.5
        
        # Instance ID patterns (AWS, GCP, Azure)
        elif re.match(r'^i-[a-f0-9]{8,17}$', value_str):  # AWS instance
            confidence += 0.6
        elif re.match(r'^[a-z]+-[a-z0-9]+-[a-z0-9]+$', value_str):  # Cloud instance
            confidence += 0.4
        
        # Check column statistics
        if column_metadata:
            unique_ratio = column_metadata.get('unique_ratio', 0)
            if unique_ratio > 0.9:  # Highly unique values
                confidence += 0.1
        
        # Check for common non-hostname patterns
        non_hostname_patterns = [
            r'^\d+$',  # Pure numbers
            r'^[a-f0-9]{32,}$',  # Hashes
            r'@',  # Email addresses
            r'^https?://',  # URLs
            r'^\/',  # File paths
        ]
        
        for pattern in non_hostname_patterns:
            if re.search(pattern, value_str):
                confidence *= 0.5
                break
        
        return min(confidence, 1.0)
    
    def _normalize_hostname(self, hostname: Any) -> str:
        """Normalize hostname to standard format"""
        hostname = str(hostname).lower().strip()
        
        # Remove common prefixes/suffixes
        import re
        hostname = re.sub(r'^https?://', '', hostname)
        hostname = re.sub(r':\d+$', '', hostname)
        hostname = hostname.split('/')[0]
        
        # Remove trailing dots
        hostname = hostname.rstrip('.')
        
        return hostname
    
    async def classify_entities(self, hosts: Dict[str, Dict]) -> Dict[str, Dict]:
        """Classify and enrich host entities"""
        self.state['phase'] = 'classifying'
        classified = {}
        
        for hostname, host_data in hosts.items():
            # Classify the host
            classification = await self.classifier.classify_host(hostname, host_data)
            
            # Enrich with inferred attributes
            enriched_data = {
                **host_data,
                'classification': classification,
                'entity_type': classification['type'],
                'sub_type': classification.get('sub_type'),
                'environment': self._infer_environment(hostname, host_data),
                'datacenter': self._infer_datacenter(hostname, host_data),
                'os_type': self._infer_os_type(hostname, host_data),
                'application': self._infer_application(hostname, host_data),
                'owner': self._infer_owner(hostname, host_data),
                'criticality': self._calculate_criticality(hostname, host_data, classification)
            }
            
            # Deduplicate and clean attributes
            for attr_name, values in enriched_data.get('attributes', {}).items():
                if isinstance(values, list):
                    # Keep unique values
                    unique_values = []
                    seen = set()
                    for v in values:
                        v_str = str(v).lower() if v else ''
                        if v_str and v_str not in seen:
                            unique_values.append(v)
                            seen.add(v_str)
                    enriched_data['attributes'][attr_name] = unique_values[:10]  # Limit to 10
            
            classified[hostname] = enriched_data
        
        self.state['classified_entities'] = classified
        return classified
    
    def _infer_environment(self, hostname: str, host_data: Dict) -> str:
        """Infer environment from hostname and attributes"""
        hostname_lower = hostname.lower()
        
        # Check hostname patterns
        env_patterns = {
            'production': ['prod', 'prd', 'production', 'live'],
            'staging': ['stage', 'stg', 'staging', 'uat'],
            'development': ['dev', 'develop', 'development'],
            'testing': ['test', 'tst', 'qa'],
            'disaster_recovery': ['dr', 'disaster', 'recovery'],
            'sandbox': ['sandbox', 'sbx', 'demo', 'poc']
        }
        
        for env, patterns in env_patterns.items():
            for pattern in patterns:
                if pattern in hostname_lower:
                    return env
        
        # Check attributes
        attrs = host_data.get('attributes', {})
        for attr_name, values in attrs.items():
            if 'env' in attr_name.lower():
                if values and isinstance(values, list):
                    return str(values[0]).lower()
        
        return 'unknown'
    
    def _infer_datacenter(self, hostname: str, host_data: Dict) -> str:
        """Infer datacenter/region from hostname and attributes"""
        hostname_lower = hostname.lower()
        
        # Common datacenter/region patterns
        dc_patterns = {
            'us-east-1': ['use1', 'useast1', 'virginia', 'iad'],
            'us-west-1': ['usw1', 'uswest1', 'california', 'sfo'],
            'us-west-2': ['usw2', 'uswest2', 'oregon', 'pdx'],
            'eu-west-1': ['euw1', 'euwest1', 'ireland', 'dub'],
            'eu-central-1': ['euc1', 'eucentral1', 'frankfurt', 'fra'],
            'ap-southeast-1': ['apse1', 'singapore', 'sin'],
            'ap-northeast-1': ['apne1', 'tokyo', 'nrt']
        }
        
        for dc, patterns in dc_patterns.items():
            for pattern in patterns:
                if pattern in hostname_lower:
                    return dc
        
        # Check IP address for private ranges
        attrs = host_data.get('attributes', {})
        for attr_name, values in attrs.items():
            if 'ip' in attr_name.lower() and values:
                ip = str(values[0])
                if ip.startswith('10.'):
                    octet2 = int(ip.split('.')[1])
                    if octet2 < 50:
                        return 'us-east-1'
                    elif octet2 < 100:
                        return 'us-west-2'
                    elif octet2 < 150:
                        return 'eu-west-1'
        
        return 'unknown'
    
    def _infer_os_type(self, hostname: str, host_data: Dict) -> str:
        """Infer operating system type"""
        hostname_lower = hostname.lower()
        attrs = host_data.get('attributes', {})
        
        # Check hostname patterns
        os_patterns = {
            'windows': ['win', 'windows', 'ws', 'w2k'],
            'linux': ['linux', 'lnx', 'ubuntu', 'centos', 'rhel', 'debian'],
            'macos': ['mac', 'osx', 'darwin'],
            'aix': ['aix'],
            'solaris': ['sol', 'solaris', 'sunos'],
            'freebsd': ['bsd', 'freebsd']
        }
        
        for os_type, patterns in os_patterns.items():
            for pattern in patterns:
                if pattern in hostname_lower:
                    return os_type
        
        # Check attributes
        for attr_name, values in attrs.items():
            attr_lower = attr_name.lower()
            if 'os' in attr_lower or 'operating' in attr_lower:
                if values:
                    value_lower = str(values[0]).lower()
                    for os_type, patterns in os_patterns.items():
                        for pattern in patterns:
                            if pattern in value_lower:
                                return os_type
        
        # Default based on naming convention
        if '-' in hostname:
            return 'linux'  # Linux servers often use hyphens
        elif hostname.startswith('srv') or hostname.startswith('server'):
            return 'windows'  # Windows naming convention
        
        return 'linux'  # Default to linux
    
    def _infer_application(self, hostname: str, host_data: Dict) -> str:
        """Infer application from hostname and attributes"""
        hostname_lower = hostname.lower()
        
        # Common application patterns
        app_patterns = {
            'web': ['web', 'www', 'nginx', 'apache', 'httpd'],
            'database': ['db', 'database', 'mysql', 'postgres', 'oracle', 'mongo', 'redis'],
            'cache': ['cache', 'redis', 'memcache', 'varnish'],
            'queue': ['queue', 'rabbit', 'kafka', 'mq', 'sqs'],
            'api': ['api', 'rest', 'graphql', 'gateway'],
            'mail': ['mail', 'smtp', 'imap', 'pop', 'exchange'],
            'dns': ['dns', 'bind', 'named'],
            'load_balancer': ['lb', 'loadbalancer', 'haproxy', 'elb', 'alb'],
            'monitoring': ['monitor', 'nagios', 'zabbix', 'prometheus', 'grafana'],
            'logging': ['log', 'elastic', 'logstash', 'splunk', 'fluentd'],
            'storage': ['storage', 'nas', 'san', 'nfs', 'ceph'],
            'backup': ['backup', 'bkp', 'vault'],
            'ci_cd': ['jenkins', 'gitlab', 'bamboo', 'travis', 'circleci'],
            'container': ['docker', 'k8s', 'kubernetes', 'swarm', 'ecs']
        }
        
        for app, patterns in app_patterns.items():
            for pattern in patterns:
                if pattern in hostname_lower:
                    return app
        
        # Check attributes
        attrs = host_data.get('attributes', {})
        for attr_name, values in attrs.items():
            if 'app' in attr_name.lower() or 'service' in attr_name.lower():
                if values:
                    return str(values[0])
        
        return 'unknown'
    
    def _infer_owner(self, hostname: str, host_data: Dict) -> str:
        """Infer owner/team from hostname and attributes"""
        attrs = host_data.get('attributes', {})
        
        # Check owner-related attributes
        owner_attrs = ['owner', 'team', 'department', 'contact', 'managed_by']
        for attr_name, values in attrs.items():
            attr_lower = attr_name.lower()
            for owner_attr in owner_attrs:
                if owner_attr in attr_lower and values:
                    return str(values[0])
        
        # Infer from hostname patterns
        hostname_lower = hostname.lower()
        
        # Team patterns
        team_patterns = {
            'infrastructure': ['infra', 'ops', 'sre'],
            'development': ['dev', 'eng'],
            'database': ['db', 'dba'],
            'security': ['sec', 'security'],
            'network': ['net', 'network'],
            'data': ['data', 'analytics', 'bi']
        }
        
        for team, patterns in team_patterns.items():
            for pattern in patterns:
                if pattern in hostname_lower:
                    return team
        
        return 'unknown'
    
    def _calculate_criticality(self, hostname: str, host_data: Dict, classification: Dict) -> str:
        """Calculate host criticality level"""
        score = 0
        
        # Environment factor
        env = host_data.get('environment', 'unknown')
        if env == 'production':
            score += 40
        elif env == 'disaster_recovery':
            score += 35
        elif env == 'staging':
            score += 20
        elif env == 'development':
            score += 10
        
        # Application type factor
        app = host_data.get('application', 'unknown')
        critical_apps = ['database', 'api', 'load_balancer', 'queue', 'cache']
        if app in critical_apps:
            score += 30
        elif app in ['web', 'mail']:
            score += 20
        elif app in ['monitoring', 'logging']:
            score += 15
        
        # Classification confidence
        if classification.get('confidence', 0) > 0.9:
            score += 10
        
        # Number of sources (indicates importance)
        n_sources = len(host_data.get('sources', []))
        if n_sources > 5:
            score += 20
        elif n_sources > 2:
            score += 10
        
        # Determine criticality level
        if score >= 70:
            return 'critical'
        elif score >= 50:
            return 'high'
        elif score >= 30:
            return 'medium'
        else:
            return 'low'
    
    async def map_relationships(self, entities: Dict[str, Dict]) -> List[Dict]:
        """Map relationships between entities"""
        self.state['phase'] = 'mapping_relationships'
        relationships = []
        
        # Build relationship graph
        for hostname1, entity1 in entities.items():
            for hostname2, entity2 in entities.items():
                if hostname1 >= hostname2:  # Avoid duplicates
                    continue
                
                # Check for relationships
                rel_type, confidence = self._detect_relationship(entity1, entity2)
                
                if rel_type and confidence > self.config.get('relationship_confidence_threshold', 0.5):
                    relationships.append({
                        'source': hostname1,
                        'target': hostname2,
                        'type': rel_type,
                        'confidence': confidence,
                        'discovered_at': datetime.now().isoformat()
                    })
        
        # Add relationships from shared attributes
        attr_relationships = self._find_attribute_relationships(entities)
        relationships.extend(attr_relationships)
        
        self.state['relationships'] = relationships
        self.state['statistics']['relationships'] = len(relationships)
        
        logger.info(f"Mapped {len(relationships)} relationships")
        return relationships
    
    def _detect_relationship(self, entity1: Dict, entity2: Dict) -> tuple:
        """Detect relationship type between two entities"""
        
        # Same environment
        if entity1.get('environment') == entity2.get('environment') != 'unknown':
            
            # Same application
            if entity1.get('application') == entity2.get('application') != 'unknown':
                return 'same_application', 0.9
            
            # Load balancer -> Web server
            if entity1.get('application') == 'load_balancer' and entity2.get('application') == 'web':
                return 'load_balances', 0.8
            if entity2.get('application') == 'load_balancer' and entity1.get('application') == 'web':
                return 'load_balanced_by', 0.8
            
            # Web server -> Database
            if entity1.get('application') == 'web' and entity2.get('application') == 'database':
                return 'connects_to', 0.7
            if entity2.get('application') == 'web' and entity1.get('application') == 'database':
                return 'connected_from', 0.7
            
            # Same datacenter
            if entity1.get('datacenter') == entity2.get('datacenter') != 'unknown':
                return 'same_datacenter', 0.6
        
        # Parent-child based on hostname
        hostname1 = entity1.get('hostname', '')
        hostname2 = entity2.get('hostname', '')
        
        if hostname1 and hostname2:
            if hostname1.startswith(hostname2.split('.')[0] + '-'):
                return 'child_of', 0.7
            if hostname2.startswith(hostname1.split('.')[0] + '-'):
                return 'parent_of', 0.7
        
        return None, 0.0
    
    def _find_attribute_relationships(self, entities: Dict[str, Dict]) -> List[Dict]:
        """Find relationships based on shared attributes"""
        relationships = []
        attribute_index = defaultdict(set)
        
        # Build index of entities by attribute values
        for hostname, entity in entities.items():
            for attr_name, values in entity.get('attributes', {}).items():
                if isinstance(values, list):
                    for value in values[:5]:  # Limit to first 5 values
                        if value:
                            key = f"{attr_name}:{str(value).lower()}"
                            attribute_index[key].add(hostname)
        
        # Find entities sharing rare attributes
        for key, hostnames in attribute_index.items():
            if 2 <= len(hostnames) <= 10:  # Rare but not unique
                attr_name = key.split(':')[0]
                
                # Determine relationship type based on attribute
                rel_type = 'shares_attribute'
                confidence = 0.4
                
                if 'cluster' in attr_name.lower():
                    rel_type = 'same_cluster'
                    confidence = 0.8
                elif 'vlan' in attr_name.lower() or 'subnet' in attr_name.lower():
                    rel_type = 'same_network'
                    confidence = 0.7
                elif 'owner' in attr_name.lower() or 'team' in attr_name.lower():
                    rel_type = 'same_owner'
                    confidence = 0.6
                
                # Create relationships between all pairs
                hostnames_list = list(hostnames)
                for i in range(len(hostnames_list)):
                    for j in range(i + 1, len(hostnames_list)):
                        relationships.append({
                            'source': hostnames_list[i],
                            'target': hostnames_list[j],
                            'type': rel_type,
                            'confidence': confidence,
                            'via_attribute': attr_name,
                            'discovered_at': datetime.now().isoformat()
                        })
        
        return relationships
    
    async def build_cmdb(self, entities: Dict[str, Dict], relationships: List[Dict]):
        """Build the CMDB database"""
        self.state['phase'] = 'building_cmdb'
        
        # Initialize database
        await self.cmdb_builder.initialize()
        
        # Create schema based on discovered attributes
        all_attributes = set()
        for entity in entities.values():
            all_attributes.update(entity.get('attributes', {}).keys())
        
        await self.cmdb_builder.create_schema(list(all_attributes))
        
        # Insert entities
        await self.cmdb_builder.insert_entities(list(entities.values()))
        
        # Insert relationships
        await self.cmdb_builder.insert_relationships(relationships)
        
        # Create indexes for performance
        await self.cmdb_builder.create_indexes()
        
        # Calculate and store statistics
        stats = self.calculate_statistics(entities, relationships)
        await self.cmdb_builder.store_statistics(stats)
        
        self.state['phase'] = 'completed'
        self.state['end_time'] = datetime.now().isoformat()
        
        logger.info(f"CMDB built successfully with {len(entities)} entities and {len(relationships)} relationships")
    
    def calculate_statistics(self, entities: Dict[str, Dict], relationships: List[Dict]) -> Dict:
        """Calculate comprehensive statistics"""
        stats = {
            'start_time': self.start_time.isoformat(),
            'end_time': datetime.now().isoformat(),
            'duration': str(datetime.now() - self.start_time),
            **self.state['statistics']
        }
        
        # Entity statistics
        stats['total_entities'] = len(entities)
        stats['environments'] = len(set(e.get('environment') for e in entities.values()))
        stats['datacenters'] = len(set(e.get('datacenter') for e in entities.values()))
        stats['applications'] = len(set(e.get('application') for e in entities.values()))
        stats['os_types'] = len(set(e.get('os_type') for e in entities.values()))
        
        # Criticality distribution
        criticality_dist = defaultdict(int)
        for entity in entities.values():
            criticality_dist[entity.get('criticality', 'unknown')] += 1
        stats['criticality_distribution'] = dict(criticality_dist)
        
        # Relationship statistics
        stats['total_relationships'] = len(relationships)
        rel_type_dist = defaultdict(int)
        for rel in relationships:
            rel_type_dist[rel['type']] += 1
        stats['relationship_types'] = dict(rel_type_dist)
        
        # Quality metrics
        entities_with_env = sum(1 for e in entities.values() if e.get('environment') != 'unknown')
        entities_with_app = sum(1 for e in entities.values() if e.get('application') != 'unknown')
        entities_with_owner = sum(1 for e in entities.values() if e.get('owner') != 'unknown')
        
        stats['classification_accuracy'] = (entities_with_env + entities_with_app + entities_with_owner) / (3 * len(entities)) if entities else 0
        
        # Average confidence scores
        avg_entity_confidence = sum(e.get('confidence', 0) for e in entities.values()) / len(entities) if entities else 0
        avg_rel_confidence = sum(r.get('confidence', 0) for r in relationships) / len(relationships) if relationships else 0
        
        stats['avg_entity_confidence'] = avg_entity_confidence
        stats['avg_relationship_confidence'] = avg_rel_confidence
        stats['quality_score'] = (avg_entity_confidence + stats['classification_accuracy']) / 2
        
        return stats
    
    async def load_cmdb(self) -> Dict:
        """Load existing CMDB data"""
        return await self.cmdb_builder.load_all_data()
    
    async def analyze_data_quality(self, cmdb_data: Dict) -> Dict:
        """Analyze CMDB data quality"""
        return self.quality_analyzer.analyze(cmdb_data)
    
    async def detect_anomalies(self, cmdb_data: Dict) -> List[Dict]:
        """Detect anomalies in CMDB data"""
        return self.quality_analyzer.detect_anomalies(cmdb_data)
    
    async def generate_insights(self, cmdb_data: Dict) -> List[str]:
        """Generate insights from CMDB data"""
        return self.quality_analyzer.generate_insights(cmdb_data)
    
    def save_checkpoint(self):
        """Save current state to checkpoint file"""
        try:
            with open(self.checkpoint_file, 'wb') as f:
                pickle.dump(self.state, f)
            logger.info(f"Checkpoint saved to {self.checkpoint_file}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
    
    def load_checkpoint(self) -> bool:
        """Load state from checkpoint file"""
        if not self.checkpoint_file.exists():
            return False
        
        try:
            with open(self.checkpoint_file, 'rb') as f:
                self.state = pickle.load(f)
            logger.info(f"Checkpoint loaded from {self.checkpoint_file}")
            logger.info(f"Resuming from phase: {self.state.get('phase')}")
            return True
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return False
    
    def get_statistics(self) -> Dict:
        """Get current statistics"""
        return self.calculate_statistics(
            self.state.get('classified_entities', {}),
            self.state.get('relationships', [])
        )