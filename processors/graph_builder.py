import networkx as nx
from typing import Dict, List, Set, Tuple
import numpy as np
from collections import defaultdict
import community.community_louvain as community_louvain

class GraphBuilder:
    def __init__(self):
        self.graph = nx.Graph()
        self.entity_attributes = {}
        self.relationship_types = {
            'same_entity': 1.0,
            'same_network': 0.8,
            'same_datacenter': 0.6,
            'same_application': 0.5,
            'related': 0.3
        }
    
    def build_entity_graph(self, entities: Dict) -> nx.Graph:
        self.graph.clear()
        
        for entity_id, entity_data in entities.items():
            self.graph.add_node(entity_id, **entity_data)
            self.entity_attributes[entity_id] = entity_data.get('attributes', {})
        
        self._add_direct_relationships(entities)
        self._add_attribute_relationships()
        self._add_network_relationships()
        
        return self.graph
    
    def _add_direct_relationships(self, entities: Dict):
        for entity_id, entity_data in entities.items():
            raw_forms = entity_data.get('raw_forms', set())
            
            for other_id, other_data in entities.items():
                if entity_id >= other_id:
                    continue
                
                other_forms = other_data.get('raw_forms', set())
                
                if raw_forms & other_forms:
                    self.graph.add_edge(
                        entity_id,
                        other_id,
                        weight=self.relationship_types['same_entity'],
                        type='same_entity'
                    )
    
    def _add_attribute_relationships(self):
        attribute_index = defaultdict(set)
        
        for entity_id, attributes in self.entity_attributes.items():
            for attr_name, attr_values in attributes.items():
                if isinstance(attr_values, list):
                    for value in attr_values:
                        key = f"{attr_name}:{str(value).lower()}"
                        attribute_index[key].add(entity_id)
                else:
                    key = f"{attr_name}:{str(attr_values).lower()}"
                    attribute_index[key].add(entity_id)
        
        for key, entity_set in attribute_index.items():
            if len(entity_set) > 1:
                entity_list = list(entity_set)
                
                attr_name = key.split(':')[0]
                relationship_type = self._infer_relationship_type(attr_name)
                weight = self.relationship_types.get(relationship_type, 0.3)
                
                for i in range(len(entity_list)):
                    for j in range(i + 1, len(entity_list)):
                        if not self.graph.has_edge(entity_list[i], entity_list[j]):
                            self.graph.add_edge(
                                entity_list[i],
                                entity_list[j],
                                weight=weight,
                                type=relationship_type,
                                via_attribute=attr_name
                            )
    
    def _add_network_relationships(self):
        for node in self.graph.nodes():
            if self._is_ip_address(node):
                network = self._get_network_prefix(node)
                
                for other_node in self.graph.nodes():
                    if node != other_node and self._is_ip_address(other_node):
                        other_network = self._get_network_prefix(other_node)
                        
                        if network == other_network:
                            if not self.graph.has_edge(node, other_node):
                                self.graph.add_edge(
                                    node,
                                    other_node,
                                    weight=self.relationship_types['same_network'],
                                    type='same_network'
                                )
    
    def find_communities(self, graph: nx.Graph = None) -> List[Set[str]]:
        if graph is None:
            graph = self.graph
        
        if len(graph) == 0:
            return []
        
        partition = community_louvain.best_partition(graph)
        
        communities = defaultdict(set)
        for node, community_id in partition.items():
            communities[community_id].add(node)
        
        return list(communities.values())
    
    def find_important_nodes(self, top_n: int = 20) -> List[Tuple[str, float]]:
        if len(self.graph) == 0:
            return []
        
        pagerank = nx.pagerank(self.graph, weight='weight')
        
        betweenness = nx.betweenness_centrality(self.graph, weight='weight')
        
        degree_centrality = nx.degree_centrality(self.graph)
        
        combined_scores = {}
        for node in self.graph.nodes():
            combined_scores[node] = (
                pagerank.get(node, 0) * 0.4 +
                betweenness.get(node, 0) * 0.3 +
                degree_centrality.get(node, 0) * 0.3
            )
        
        sorted_nodes = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_nodes[:top_n]
    
    def find_paths(self, source: str, target: str, max_length: int = 5) -> List[List[str]]:
        if source not in self.graph or target not in self.graph:
            return []
        
        try:
            all_paths = list(nx.all_simple_paths(
                self.graph,
                source,
                target,
                cutoff=max_length
            ))
            
            all_paths.sort(key=len)
            
            return all_paths[:10]
            
        except nx.NetworkXNoPath:
            return []
    
    def get_subgraph(self, nodes: List[str], radius: int = 1) -> nx.Graph:
        expanded_nodes = set(nodes)
        
        for node in nodes:
            if node in self.graph:
                for neighbor in nx.single_source_shortest_path_length(
                    self.graph, node, cutoff=radius
                ):
                    expanded_nodes.add(neighbor)
        
        return self.graph.subgraph(expanded_nodes)
    
    def _infer_relationship_type(self, attribute_name: str) -> str:
        attr_lower = attribute_name.lower()
        
        if any(term in attr_lower for term in ['datacenter', 'dc', 'location', 'site']):
            return 'same_datacenter'
        elif any(term in attr_lower for term in ['app', 'application', 'service']):
            return 'same_application'
        elif any(term in attr_lower for term in ['network', 'subnet', 'vlan']):
            return 'same_network'
        else:
            return 'related'
    
    def _is_ip_address(self, value: str) -> bool:
        import re
        ipv4_pattern = r'^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$'
        return bool(re.match(ipv4_pattern, value))
    
    def _get_network_prefix(self, ip: str) -> str:
        parts = ip.split('.')
        if len(parts) >= 3:
            return '.'.join(parts[:3])
        return ip