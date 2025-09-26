"""
Intelligent Classifier - Classifies and categorizes discovered entities
"""

import re
from typing import Dict, List, Any, Optional
from collections import Counter
import logging

logger = logging.getLogger(__name__)

class IntelligentClassifier:
    """Classifies entities using pattern matching and heuristics"""
    
    def __init__(self):
        self.classification_rules = self._build_classification_rules()
        self.pattern_cache = {}
    
    def _build_classification_rules(self) -> Dict:
        """Build classification rules and patterns"""
        return {
            'entity_types': {
                'physical_server': {
                    'patterns': [r'^srv-', r'^server-', r'^phy-'],
                    'keywords': ['physical', 'bare-metal', 'hardware'],
                    'confidence_boost': 0.2
                },
                'virtual_machine': {
                    'patterns': [r'^vm-', r'^virt-', r'^guest-'],
                    'keywords': ['virtual', 'vm', 'vmware', 'hyperv', 'kvm'],
                    'confidence_boost': 0.2
                },
                'container': {
                    'patterns': [r'^pod-', r'^container-', r'^docker-'],
                    'keywords': ['docker', 'kubernetes', 'k8s', 'container', 'pod'],
                    'confidence_boost': 0.25
                },
                'cloud_instance': {
                    'patterns': [r'^i-[a-f0-9]{8,}', r'^[a-z]+-[a-z0-9]+-[a-z0-9]+'],
                    'keywords': ['ec2', 'instance', 'aws', 'azure', 'gcp'],
                    'confidence_boost': 0.3
                },
                'network_device': {
                    'patterns': [r'^sw-', r'^switch-', r'^router-', r'^fw-'],
                    'keywords': ['switch', 'router', 'firewall', 'gateway', 'vpn'],
                    'confidence_boost': 0.25
                },
                'storage_device': {
                    'patterns': [r'^nas-', r'^san-', r'^storage-'],
                    'keywords': ['storage', 'nas', 'san', 'netapp', 'emc'],
                    'confidence_boost': 0.2
                },
                'load_balancer': {
                    'patterns': [r'^lb-', r'^alb-', r'^elb-', r'^haproxy-'],
                    'keywords': ['loadbalancer', 'lb', 'haproxy', 'nginx', 'f5'],
                    'confidence_boost': 0.3
                }
            },
            'sub_types': {
                'web_server': ['apache', 'nginx', 'iis', 'httpd', 'web'],
                'app_server': ['tomcat', 'jboss', 'websphere', 'app'],
                'database_server': ['mysql', 'postgres', 'oracle', 'mssql', 'mongodb', 'db'],
                'cache_server': ['redis', 'memcache', 'varnish', 'cache'],
                'message_queue': ['rabbitmq', 'kafka', 'activemq', 'sqs', 'queue'],
                'monitoring': ['nagios', 'zabbix', 'prometheus', 'grafana', 'monitor'],
                'ci_cd': ['jenkins', 'gitlab', 'bamboo', 'travis', 'circleci'],
                'security': ['firewall', 'ids', 'ips', 'waf', 'security']
            }
        }
    
    async def classify_host(self, hostname: str, host_data: Dict) -> Dict:
        """Classify a host entity"""
        classification = {
            'type': 'unknown',
            'sub_type': 'unknown',
            'confidence': 0.0,
            'classification_method': 'unknown',
            'tags': []
        }
        
        # Classify entity type
        entity_type, type_confidence = self._classify_entity_type(hostname, host_data)
        classification['type'] = entity_type
        
        # Classify sub-type
        sub_type = self._classify_sub_type(hostname, host_data)
        classification['sub_type'] = sub_type
        
        # Calculate overall confidence
        attribute_confidence = self._calculate_attribute_confidence(host_data)
        source_confidence = self._calculate_source_confidence(host_data)
        
        classification['confidence'] = min(
            (type_confidence + attribute_confidence + source_confidence) / 3,
            1.0
        )
        
        # Determine classification method
        if type_confidence > 0.8:
            classification['classification_method'] = 'pattern_matching'
        elif attribute_confidence > 0.7:
            classification['classification_method'] = 'attribute_analysis'
        elif source_confidence > 0.6:
            classification['classification_method'] = 'source_correlation'
        else:
            classification['classification_method'] = 'heuristic'
        
        # Add tags
        classification['tags'] = self._generate_tags(hostname, host_data, classification)
        
        return classification
    
    def _classify_entity_type(self, hostname: str, host_data: Dict) -> tuple:
        """Classify the entity type"""
        hostname_lower = hostname.lower()
        scores = {}
        
        for entity_type, rules in self.classification_rules['entity_types'].items():
            score = 0.0
            
            # Check hostname patterns
            for pattern in rules['patterns']:
                if re.match(pattern, hostname_lower):
                    score += 0.4
                    break
            
            # Check keywords in hostname
            for keyword in rules['keywords']:
                if keyword in hostname_lower:
                    score += 0.2
            
            # Check attributes
            attrs = host_data.get('attributes', {})
            for attr_name, values in attrs.items():
                attr_lower = attr_name.lower()
                
                # Check attribute names
                for keyword in rules['keywords']:
                    if keyword in attr_lower:
                        score += 0.1
                
                # Check attribute values
                if values and isinstance(values, list):
                    for value in values[:5]:
                        value_lower = str(value).lower()
                        for keyword in rules['keywords']:
                            if keyword in value_lower:
                                score += 0.1
                                break
            
            # Apply confidence boost
            score += rules.get('confidence_boost', 0)
            scores[entity_type] = min(score, 1.0)
        
        # Select best match
        if scores:
            best_type = max(scores, key=scores.get)
            confidence = scores[best_type]
            
            # Default to virtual_machine if confidence is too low
            if confidence < 0.3:
                return 'virtual_machine', 0.5
            
            return best_type, confidence
        
        return 'virtual_machine', 0.4
    
    def _classify_sub_type(self, hostname: str, host_data: Dict) -> str:
        """Classify the sub-type (application role)"""
        hostname_lower = hostname.lower()
        
        for sub_type, keywords in self.classification_rules['sub_types'].items():
            for keyword in keywords:
                if keyword in hostname_lower:
                    return sub_type
            
            # Check attributes
            attrs = host_data.get('attributes', {})
            for attr_name, values in attrs.items():
                attr_lower = attr_name.lower()
                
                for keyword in keywords:
                    if keyword in attr_lower:
                        return sub_type
                
                if values and isinstance(values, list):
                    for value in values[:5]:
                        value_lower = str(value).lower()
                        for keyword in keywords:
                            if keyword in value_lower:
                                return sub_type
        
        # Check for common patterns
        if 'app' in hostname_lower or 'application' in hostname_lower:
            return 'app_server'
        elif 'web' in hostname_lower or 'www' in hostname_lower:
            return 'web_server'
        elif 'db' in hostname_lower or 'database' in hostname_lower:
            return 'database_server'
        elif 'api' in hostname_lower:
            return 'app_server'
        
        return 'general_purpose'
    
    def _calculate_attribute_confidence(self, host_data: Dict) -> float:
        """Calculate confidence based on attribute completeness"""
        attrs = host_data.get('attributes', {})
        
        if not attrs:
            return 0.3
        
        # Important attributes
        important_attrs = ['ip', 'environment', 'datacenter', 'os', 'application', 'owner']
        found_count = 0
        
        for attr_name in attrs.keys():
            attr_lower = attr_name.lower()
            for important in important_attrs:
                if important in attr_lower:
                    found_count += 1
                    break
        
        # Calculate completeness score
        completeness = found_count / len(important_attrs)
        
        # Bonus for having many attributes
        attr_count_bonus = min(len(attrs) / 20, 0.3)
        
        return min(completeness + attr_count_bonus, 1.0)
    
    def _calculate_source_confidence(self, host_data: Dict) -> float:
        """Calculate confidence based on data sources"""
        sources = host_data.get('sources', [])
        
        if not sources:
            return 0.2
        
        # More sources = higher confidence
        source_count_score = min(len(sources) / 5, 0.5)
        
        # Average confidence from sources
        if sources:
            avg_confidence = sum(s.get('confidence', 0) for s in sources) / len(sources)
        else:
            avg_confidence = 0
        
        # Diverse sources boost confidence
        unique_source_names = len(set(s.get('source', '') for s in sources))
        diversity_score = min(unique_source_names / 3, 0.3)
        
        return min(source_count_score + avg_confidence * 0.5 + diversity_score, 1.0)
    
    def _generate_tags(self, hostname: str, host_data: Dict, classification: Dict) -> List[str]:
        """Generate descriptive tags for the entity"""
        tags = []
        
        # Add type tags
        if classification['type'] != 'unknown':
            tags.append(classification['type'])
        if classification['sub_type'] != 'unknown':
            tags.append(classification['sub_type'])
        
        # Environment tags
        env = host_data.get('environment')
        if env and env != 'unknown':
            tags.append(f"env:{env}")
        
        # Datacenter tags
        dc = host_data.get('datacenter')
        if dc and dc != 'unknown':
            tags.append(f"dc:{dc}")
        
        # OS tags
        os_type = host_data.get('os_type')
        if os_type and os_type != 'unknown':
            tags.append(f"os:{os_type}")
        
        # Application tags
        app = host_data.get('application')
        if app and app != 'unknown':
            tags.append(f"app:{app}")
        
        # Criticality tags
        criticality = host_data.get('criticality')
        if criticality:
            tags.append(f"criticality:{criticality}")
        
        # Owner tags
        owner = host_data.get('owner')
        if owner and owner != 'unknown':
            tags.append(f"owner:{owner}")
        
        # Pattern-based tags
        hostname_lower = hostname.lower()
        
        if 'test' in hostname_lower or 'qa' in hostname_lower:
            tags.append('non-production')
        elif 'prod' in hostname_lower or 'prd' in hostname_lower:
            tags.append('production')
        
        if 'backup' in hostname_lower or 'bkp' in hostname_lower:
            tags.append('backup')
        
        if 'dr' in hostname_lower:
            tags.append('disaster-recovery')
        
        if 'dev' in hostname_lower:
            tags.append('development')
        
        if 'temp' in hostname_lower or 'tmp' in hostname_lower:
            tags.append('temporary')
        
        # Cloud provider tags
        if re.match(r'^i-[a-f0-9]{8,}', hostname_lower):
            tags.append('aws')
        elif 'azure' in hostname_lower:
            tags.append('azure')
        elif 'gcp' in hostname_lower or 'gce' in hostname_lower:
            tags.append('gcp')
        
        # Remove duplicates while preserving order
        seen = set()
        unique_tags = []
        for tag in tags:
            if tag not in seen:
                seen.add(tag)
                unique_tags.append(tag)
        
        return unique_tags