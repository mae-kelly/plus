"""
Relationship Mapper - Maps and discovers relationships between entities
"""

import re
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

class RelationshipMapper:
    """Maps relationships between discovered entities"""
    
    def __init__(self):
        self.relationship_types = {
            'same_application': {
                'description': 'Part of the same application stack',
                'bidirectional': True,
                'strength': 0.9
            },
            'load_balances': {
                'description': 'Load balancer distributes traffic to',
                'bidirectional': False,
                'strength': 0.8
            },
            'load_balanced_by': {
                'description': 'Receives traffic from load balancer',
                'bidirectional': False,
                'strength': 0.8
            },
            'connects_to': {
                'description': 'Has connection to',
                'bidirectional': False,
                'strength': 0.7
            },
            'connected_from': {
                'description': 'Receives connections from',
                'bidirectional': False,
                'strength': 0.7
            },
            'same_datacenter': {
                'description': 'Located in same datacenter',
                'bidirectional': True,
                'strength': 0.6
            },
            'same_network': {
                'description': 'On the same network segment',
                'bidirectional': True,
                'strength': 0.7
            },
            'same_environment': {
                'description': 'In the same environment tier',
                'bidirectional': True,
                'strength': 0.5
            },
            'parent_of': {
                'description': 'Parent entity of',
                'bidirectional': False,
                'strength': 0.8
            },
            'child_of': {
                'description': 'Child entity of',
                'bidirectional': False,
                'strength': 0.8
            },
            'same_cluster': {
                'description': 'Part of the same cluster',
                'bidirectional': True,
                'strength': 0.9
            },
            'same_owner': {
                'description': 'Managed by same team/owner',
                'bidirectional': True,
                'strength': 0.5
            },
            'shares_attribute': {
                'description': 'Shares common attribute',
                'bidirectional': True,
                'strength': 0.4
            },
            'depends_on': {
                'description': 'Has dependency on',
                'bidirectional': False,
                'strength': 0.8
            },
            'replicated_from': {
                'description': 'Is replica of',
                'bidirectional': False,
                'strength': 0.7
            },
            'backs_up': {
                'description': 'Provides backup for',
                'bidirectional': False,
                'strength': 0.6
            },
            'monitors': {
                'description': 'Monitors the health of',
                'bidirectional': False,
                'strength': 0.5
            },
            'monitored_by': {
                'description': 'Health monitored by',
                'bidirectional': False,
                'strength': 0.5
            }
        }
        
        # Application connection patterns
        self.app_connection_patterns = {
            ('web', 'database'): 'connects_to',
            ('api', 'database'): 'connects_to',
            ('app', 'database'): 'connects_to',
            ('web', 'cache'): 'connects_to',
            ('api', 'cache'): 'connects_to',
            ('app', 'cache'): 'connects_to',
            ('web', 'queue'): 'connects_to',
            ('api', 'queue'): 'connects_to',
            ('app', 'queue'): 'connects_to',
            ('load_balancer', 'web'): 'load_balances',
            ('load_balancer', 'api'): 'load_balances',
            ('load_balancer', 'app'): 'load_balances',
            ('monitoring', 'web'): 'monitors',
            ('monitoring', 'database'): 'monitors',
            ('monitoring', 'api'): 'monitors',
            ('backup', 'database'): 'backs_up',
            ('backup', 'storage'): 'backs_up'
        }
    
    def map_relationships(self, entities: Dict[str, Dict]) -> List[Dict]:
        """Map all relationships between entities"""
        relationships = []
        
        # Direct relationships
        direct_rels = self._map_direct_relationships(entities)
        relationships.extend(direct_rels)
        
        # Network relationships
        network_rels = self._map_network_relationships(entities)
        relationships.extend(network_rels)
        
        # Application relationships
        app_rels = self._map_application_relationships(entities)
        relationships.extend(app_rels)
        
        # Hierarchical relationships
        hierarchy_rels = self._map_hierarchical_relationships(entities)
        relationships.extend(hierarchy_rels)
        
        # Cluster relationships
        cluster_rels = self._map_cluster_relationships(entities)
        relationships.extend(cluster_rels)
        
        # Remove duplicates
        unique_relationships = self._deduplicate_relationships(relationships)
        
        logger.info(f"Mapped {len(unique_relationships)} unique relationships from {len(relationships)} total")
        
        return unique_relationships
    
    def _map_direct_relationships(self, entities: Dict[str, Dict]) -> List[Dict]:
        """Map direct relationships based on shared attributes"""
        relationships = []
        
        # Index entities by attribute values
        attribute_index = defaultdict(set)
        
        for hostname, entity in entities.items():
            # Index by environment
            env = entity.get('environment')
            if env and env != 'unknown':
                attribute_index[f'env:{env}'].add(hostname)
            
            # Index by datacenter
            dc = entity.get('datacenter')
            if dc and dc != 'unknown':
                attribute_index[f'dc:{dc}'].add(hostname)
            
            # Index by application
            app = entity.get('application')
            if app and app != 'unknown':
                attribute_index[f'app:{app}'].add(hostname)
            
            # Index by owner
            owner = entity.get('owner')
            if owner and owner != 'unknown':
                attribute_index[f'owner:{owner}'].add(hostname)
            
            # Index by custom attributes
            for attr_name, values in entity.get('attributes', {}).items():
                if isinstance(values, list):
                    for value in values[:5]:
                        if value:
                            key = f"{attr_name}:{str(value).lower()}"
                            attribute_index[key].add(hostname)
        
        # Create relationships based on shared attributes
        for key, hostnames in attribute_index.items():
            if len(hostnames) < 2 or len(hostnames) > 50:  # Skip unique or too common
                continue
            
            attr_type = key.split(':')[0]
            
            # Determine relationship type
            if attr_type == 'env':
                rel_type = 'same_environment'
                confidence = 0.5
            elif attr_type == 'dc':
                rel_type = 'same_datacenter'
                confidence = 0.6
            elif attr_type == 'app':
                rel_type = 'same_application'
                confidence = 0.9
            elif attr_type == 'owner':
                rel_type = 'same_owner'
                confidence = 0.5
            elif attr_type in ['cluster', 'cluster_name', 'cluster_id']:
                rel_type = 'same_cluster'
                confidence = 0.9
            else:
                rel_type = 'shares_attribute'
                confidence = 0.4
            
            # Create relationships between all pairs
            hostnames_list = list(hostnames)
            for i in range(len(hostnames_list)):
                for j in range(i + 1, len(hostnames_list)):
                    relationships.append({
                        'source': hostnames_list[i],
                        'target': hostnames_list[j],
                        'type': rel_type,
                        'confidence': confidence,
                        'via_attribute': attr_type
                    })
        
        return relationships
    
    def _map_network_relationships(self, entities: Dict[str, Dict]) -> List[Dict]:
        """Map relationships based on network topology"""
        relationships = []
        
        # Group by IP subnet
        subnet_groups = defaultdict(list)
        
        for hostname, entity in entities.items():
            # Look for IP addresses
            ip = None
            
            # Check direct IP field
            if 'ip_address' in entity:
                ip = entity['ip_address']
            else:
                # Check attributes
                for attr_name, values in entity.get('attributes', {}).items():
                    if 'ip' in attr_name.lower() and values:
                        ip = values[0] if isinstance(values, list) else values
                        break
            
            if ip and self._is_valid_ip(ip):
                subnet = self._get_subnet(ip)
                subnet_groups[subnet].append(hostname)
        
        # Create network relationships
        for subnet, hostnames in subnet_groups.items():
            if len(hostnames) < 2 or len(hostnames) > 100:
                continue
            
            for i in range(len(hostnames)):
                for j in range(i + 1, min(i + 10, len(hostnames))):  # Limit connections
                    relationships.append({
                        'source': hostnames[i],
                        'target': hostnames[j],
                        'type': 'same_network',
                        'confidence': 0.7,
                        'via_attribute': f'subnet:{subnet}'
                    })
        
        return relationships
    
    def _map_application_relationships(self, entities: Dict[str, Dict]) -> List[Dict]:
        """Map relationships based on application patterns"""
        relationships = []
        
        # Group by environment and datacenter for locality
        env_dc_groups = defaultdict(list)
        
        for hostname, entity in entities.items():
            env = entity.get('environment', 'unknown')
            dc = entity.get('datacenter', 'unknown')
            app = entity.get('application', 'unknown')
            
            key = f"{env}:{dc}"
            env_dc_groups[key].append((hostname, app))
        
        # Find application connections within same env/dc
        for group_key, members in env_dc_groups.items():
            if len(members) < 2:
                continue
            
            for i, (host1, app1) in enumerate(members):
                for j, (host2, app2) in enumerate(members[i+1:], i+1):
                    # Check for known application patterns
                    pattern_key = (app1, app2)
                    reverse_key = (app2, app1)
                    
                    if pattern_key in self.app_connection_patterns:
                        rel_type = self.app_connection_patterns[pattern_key]
                        relationships.append({
                            'source': host1,
                            'target': host2,
                            'type': rel_type,
                            'confidence': 0.8,
                            'via_attribute': f'app_pattern:{app1}->{app2}'
                        })
                    elif reverse_key in self.app_connection_patterns:
                        rel_type = self.app_connection_patterns[reverse_key]
                        relationships.append({
                            'source': host2,
                            'target': host1,
                            'type': rel_type,
                            'confidence': 0.8,
                            'via_attribute': f'app_pattern:{app2}->{app1}'
                        })
        
        return relationships
    
    def _map_hierarchical_relationships(self, entities: Dict[str, Dict]) -> List[Dict]:
        """Map parent-child relationships based on naming patterns"""
        relationships = []
        
        # Build hostname index for faster lookup
        hostnames = list(entities.keys())
        
        for host1 in hostnames:
            host1_parts = self._parse_hostname(host1)
            
            for host2 in hostnames:
                if host1 == host2:
                    continue
                
                host2_parts = self._parse_hostname(host2)
                
                # Check for parent-child patterns
                if self._is_parent_child(host1_parts, host2_parts):
                    relationships.append({
                        'source': host1,
                        'target': host2,
                        'type': 'parent_of',
                        'confidence': 0.7,
                        'via_attribute': 'hostname_hierarchy'
                    })
                elif self._is_parent_child(host2_parts, host1_parts):
                    relationships.append({
                        'source': host1,
                        'target': host2,
                        'type': 'child_of',
                        'confidence': 0.7,
                        'via_attribute': 'hostname_hierarchy'
                    })
        
        return relationships
    
    def _map_cluster_relationships(self, entities: Dict[str, Dict]) -> List[Dict]:
        """Map cluster relationships based on naming and numbering patterns"""
        relationships = []
        
        # Group by base hostname (without numbers)
        cluster_groups = defaultdict(list)
        
        for hostname in entities.keys():
            # Extract base name without numbers
            base_name = re.sub(r'\d+', '', hostname)
            cluster_groups[base_name].append(hostname)
        
        # Create cluster relationships
        for base_name, members in cluster_groups.items():
            if len(members) < 2 or len(members) > 20:  # Reasonable cluster size
                continue
            
            # Check if they have sequential numbering
            if self._has_sequential_numbering(members):
                for i in range(len(members)):
                    for j in range(i + 1, len(members)):
                        relationships.append({
                            'source': members[i],
                            'target': members[j],
                            'type': 'same_cluster',
                            'confidence': 0.8,
                            'via_attribute': f'cluster:{base_name}'
                        })
        
        return relationships
    
    def _is_valid_ip(self, ip: str) -> bool:
        """Check if string is a valid IP address"""
        parts = ip.split('.')
        if len(parts) != 4:
            return False
        
        try:
            return all(0 <= int(part) <= 255 for part in parts)
        except ValueError:
            return False
    
    def _get_subnet(self, ip: str) -> str:
        """Get subnet from IP address (assumes /24)"""
        parts = ip.split('.')
        if len(parts) >= 3:
            return '.'.join(parts[:3]) + '.0/24'
        return ip
    
    def _parse_hostname(self, hostname: str) -> Dict:
        """Parse hostname into components"""
        parts = {
            'full': hostname,
            'base': hostname.split('.')[0] if '.' in hostname else hostname,
            'domain': '.'.join(hostname.split('.')[1:]) if '.' in hostname else '',
            'segments': hostname.split('.')[0].split('-') if '-' in hostname else [hostname]
        }
        
        # Extract numbers
        numbers = re.findall(r'\d+', parts['base'])
        parts['numbers'] = numbers
        parts['base_without_numbers'] = re.sub(r'\d+', '', parts['base'])
        
        return parts
    
    def _is_parent_child(self, parent_parts: Dict, child_parts: Dict) -> bool:
        """Check if there's a parent-child relationship"""
        # Check if child hostname starts with parent base
        if child_parts['base'].startswith(parent_parts['base'] + '-'):
            return True
        
        # Check if parent has fewer segments
        if (len(parent_parts['segments']) < len(child_parts['segments']) and
            parent_parts['segments'] == child_parts['segments'][:len(parent_parts['segments'])]):
            return True
        
        return False
    
    def _has_sequential_numbering(self, hostnames: List[str]) -> bool:
        """Check if hostnames have sequential numbering pattern"""
        numbers = []
        
        for hostname in hostnames:
            # Extract all numbers from hostname
            nums = re.findall(r'\d+', hostname)
            if nums:
                try:
                    # Use the last number as sequence identifier
                    numbers.append(int(nums[-1]))
                except ValueError:
                    pass
        
        if len(numbers) < 2:
            return False
        
        # Check if numbers are somewhat sequential
        numbers.sort()
        
        # Allow gaps but check for pattern
        if len(numbers) >= 2:
            gaps = [numbers[i+1] - numbers[i] for i in range(len(numbers)-1)]
            # Most gaps should be small (1-10)
            small_gaps = sum(1 for gap in gaps if 0 < gap <= 10)
            return small_gaps > len(gaps) * 0.5
        
        return False
    
    def _deduplicate_relationships(self, relationships: List[Dict]) -> List[Dict]:
        """Remove duplicate relationships"""
        unique = {}
        
        for rel in relationships:
            # Create unique key
            source = rel['source']
            target = rel['target']
            rel_type = rel['type']
            
            # Normalize for bidirectional relationships
            if self.relationship_types.get(rel_type, {}).get('bidirectional', False):
                key = tuple(sorted([source, target]) + [rel_type])
            else:
                key = (source, target, rel_type)
            
            # Keep highest confidence version
            if key not in unique or unique[key]['confidence'] < rel['confidence']:
                unique[key] = rel
        
        return list(unique.values())
    
    def get_relationship_info(self, rel_type: str) -> Dict:
        """Get information about a relationship type"""
        return self.relationship_types.get(rel_type, {
            'description': 'Unknown relationship type',
            'bidirectional': True,
            'strength': 0.5
        })