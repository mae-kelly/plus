from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

class RelationshipAnalyzer:
    
    def __init__(self):
        self.relationships = defaultdict(lambda: defaultdict(set))
        self.statistics = {
            'total_relationships': 0,
            'relationship_types': set()
        }
        
    def analyze_host_relationships(self, hosts_data: Dict[str, Dict]) -> Dict[str, Any]:
        relationships = {
            'direct': defaultdict(set),
            'indirect': defaultdict(set),
            'clusters': []
        }
        
        # Build adjacency list for relationship graph
        adjacency = defaultdict(set)
        
        # Track which attributes connect hosts
        attribute_connections = defaultdict(lambda: defaultdict(set))
        
        for hostname, data in hosts_data.items():
            # Process associated data
            if 'associated_data' in data:
                for attr_name, attr_values in data['associated_data'].items():
                    if isinstance(attr_values, dict) and 'normalized' in attr_values:
                        for value in attr_values['normalized']:
                            if value:
                                # Track this attribute value
                                attribute_connections[attr_name][value].add(hostname)
        
        # Find relationships through shared attributes
        for attr_name, value_hosts in attribute_connections.items():
            for value, connected_hosts in value_hosts.items():
                if len(connected_hosts) > 1:
                    # These hosts share this attribute value
                    hosts_list = list(connected_hosts)
                    for i, host1 in enumerate(hosts_list):
                        for host2 in hosts_list[i+1:]:
                            adjacency[host1].add(host2)
                            adjacency[host2].add(host1)
                            relationships['indirect'][host1].add(host2)
                            relationships['indirect'][host2].add(host1)
        
        # Find clusters using connected components
        visited = set()
        
        def dfs(node, component):
            visited.add(node)
            component.add(node)
            for neighbor in adjacency[node]:
                if neighbor not in visited:
                    dfs(neighbor, component)
        
        # Find all connected components
        for hostname in hosts_data.keys():
            if hostname not in visited:
                component = set()
                dfs(hostname, component)
                if len(component) > 1:
                    relationships['clusters'].append(list(component))
        
        self.statistics['total_relationships'] = sum(
            len(hosts) for hosts in relationships['indirect'].values()
        )
        
        return relationships
        
    def find_common_attributes(self, hosts: List[str], hosts_data: Dict[str, Dict]) -> Dict[str, Set]:
        common_attrs = defaultdict(set)
        
        if not hosts:
            return common_attrs
            
        # Get attributes for first host
        first_host = hosts[0]
        if first_host not in hosts_data:
            return common_attrs
            
        first_data = hosts_data[first_host].get('associated_data', {})
        
        # Check which attributes are common
        for attr_name, attr_values in first_data.items():
            if isinstance(attr_values, dict) and 'normalized' in attr_values:
                values = set(attr_values['normalized'])
                
                # Check if other hosts have same attribute values
                is_common = True
                for other_host in hosts[1:]:
                    if other_host in hosts_data:
                        other_data = hosts_data[other_host].get('associated_data', {})
                        if attr_name in other_data:
                            other_values = set(other_data[attr_name].get('normalized', []))
                            # Find intersection
                            values = values.intersection(other_values)
                            if not values:
                                is_common = False
                                break
                        else:
                            is_common = False
                            break
                            
                if is_common and values:
                    common_attrs[attr_name] = values
                    
        return common_attrs
        
    def calculate_host_importance(self, hostname: str, hosts_data: Dict[str, Dict]) -> float:
        score = 0.0
        
        if hostname not in hosts_data:
            return score
            
        data = hosts_data[hostname]
        
        # Factor 1: Number of occurrences
        occurrences = len(data.get('occurrences', []))
        occurrence_score = min(occurrences / 100, 1.0)
        
        # Factor 2: Confidence
        confidence = max([o.get('confidence', 0) for o in data.get('occurrences', [])], default=0)
        
        # Factor 3: Number of relationships
        relationships = len(self.relationships.get(hostname, {}))
        relationship_score = min(relationships / 50, 1.0)
        
        # Factor 4: Number of attributes
        attributes = len(data.get('associated_data', {}))
        attribute_score = min(attributes / 20, 1.0)
        
        # Weighted average
        score = (
            occurrence_score * 0.3 +
            confidence * 0.3 +
            relationship_score * 0.2 +
            attribute_score * 0.2
        )
        
        return score
        
    def identify_key_attributes(self, hosts_data: Dict[str, Dict]) -> List[str]:
        # Count how often each attribute appears
        attribute_frequency = defaultdict(int)
        attribute_values = defaultdict(set)
        
        for hostname, data in hosts_data.items():
            if 'associated_data' in data:
                for attr_name, attr_vals in data['associated_data'].items():
                    attribute_frequency[attr_name] += 1
                    
                    if isinstance(attr_vals, dict) and 'normalized' in attr_vals:
                        for val in attr_vals['normalized']:
                            if val:
                                attribute_values[attr_name].add(val)
        
        # Score attributes based on frequency and uniqueness
        attribute_scores = []
        
        for attr_name, frequency in attribute_frequency.items():
            # High frequency is good
            freq_score = frequency / len(hosts_data)
            
            # Having diverse values is also good
            unique_values = len(attribute_values[attr_name])
            diversity_score = min(unique_values / 100, 1.0)
            
            # Combined score
            score = freq_score * 0.7 + diversity_score * 0.3
            
            attribute_scores.append((attr_name, score))
        
        # Sort by score
        attribute_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top attributes
        return [attr for attr, score in attribute_scores[:50]]
        
    def get_statistics(self) -> Dict[str, Any]:
        stats = {
            'total_relationships': self.statistics['total_relationships'],
            'relationship_types': list(self.statistics['relationship_types'])
        }
        
        return stats