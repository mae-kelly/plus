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
        if type