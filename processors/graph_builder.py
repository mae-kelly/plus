# processors/graph_builder.py
"""
Graph Builder - Constructs knowledge graph from discovered hosts and relationships
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import networkx as nx
from collections import defaultdict, Counter
import logging
import json

logger = logging.getLogger(__name__)

class GraphBuilder:
    """Builds comprehensive knowledge graph from CMDB data"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.graph = nx.MultiDiGraph()  # Directed graph with multiple edges
        self.node_embeddings = {}
        self.edge_embeddings = {}
        self.communities = []
        self.centrality_scores = {}
        
    async def build(self, hosts: List[Dict], relationships: List[Dict]) -> Dict:
        """Build complete knowledge graph"""
        logger.info("Building knowledge graph...")
        
        # Add nodes (hosts)
        for host in hosts:
            self._add_host_node(host)
        
        # Add edges (relationships)
        for rel in relationships:
            self._add_relationship_edge(rel)
        
        # Calculate graph metrics
        self._calculate_metrics()
        
        # Detect communities
        self._detect_communities()
        
        # Find critical paths
        critical_paths = self._find_critical_paths()
        
        # Generate embeddings
        await self._generate_embeddings()
        
        # Build hierarchy
        hierarchy = self._build_hierarchy()
        
        # Create graph summary
        graph_data = {
            'nodes': self._export_nodes(),
            'edges': self._export_edges(),
            'metrics': {
                'node_count': self.graph.number_of_nodes(),
                'edge_count': self.graph.number_of_edges(),
                'density': nx.density(self.graph),
                'is_connected': nx.is_weakly_connected(self.graph),
                'average_degree': sum(dict(self.graph.degree()).values()) / self.graph.number_of_nodes() if self.graph.number_of_nodes() > 0 else 0,
                'clustering_coefficient': nx.average_clustering(self.graph.to_undirected()),
                'communities': len(self.communities)
            },
            'communities': self._export_communities(),
            'centrality': self.centrality_scores,
            'critical_paths': critical_paths,
            'hierarchy': hierarchy
        }
        
        logger.info(f"✅ Graph built with {graph_data['metrics']['node_count']} nodes and {graph_data['metrics']['edge_count']} edges")
        
        return graph_data
    
    def _add_host_node(self, host: Dict):
        """Add host as node in graph"""
        node_id = host.get('hostname', 'unknown')
        
        # Node attributes
        attributes = {
            'type': host.get('classification', {}).get('type', 'unknown'),
            'sub_type': host.get('classification', {}).get('sub_type', 'unknown'),
            'environment': host.get('environment', 'unknown'),
            'datacenter': host.get('datacenter', 'unknown'),
            'application': host.get('application', 'unknown'),
            'owner': host.get('owner', 'unknown'),
            'confidence': host.get('confidence', 0.0),
            'quality_score': host.get('quality_score', 0.0),
            'criticality': host.get('criticality', 'medium'),
            'tags': host.get('classification', {}).get('tags', [])
        }
        
        # Add additional attributes
        for key, value in host.items():
            if key not in attributes and not isinstance(value, (dict, list)):
                attributes[key] = value
        
        self.graph.add_node(node_id, **attributes)
    
    def _add_relationship_edge(self, relationship: Dict):
        """Add relationship as edge in graph"""
        source = relationship.get('source')
        target = relationship.get('target')
        
        if not source or not target:
            return
        
        # Ensure nodes exist
        if source not in self.graph:
            self.graph.add_node(source)
        if target not in self.graph:
            self.graph.add_node(target)
        
        # Edge attributes
        attributes = {
            'type': relationship.get('type', 'related'),
            'confidence': relationship.get('confidence', 0.5),
            'via_attribute': relationship.get('via_attribute'),
            'discovered_at': relationship.get('discovered_at'),
            'metadata': relationship.get('metadata', {})
        }
        
        self.graph.add_edge(source, target, **attributes)
    
    def _calculate_metrics(self):
        """Calculate graph metrics and centrality"""
        if self.graph.number_of_nodes() == 0:
            return
        
        # Degree centrality
        self.centrality_scores['degree'] = nx.degree_centrality(self.graph)
        
        # Betweenness centrality (important for finding critical nodes)
        if self.graph.number_of_nodes() < 1000:  # Expensive for large graphs
            self.centrality_scores['betweenness'] = nx.betweenness_centrality(self.graph)
        
        # Closeness centrality
        if nx.is_weakly_connected(self.graph):
            self.centrality_scores['closeness'] = nx.closeness_centrality(self.graph)
        
        # PageRank (importance based on connections)
        self.centrality_scores['pagerank'] = nx.pagerank(self.graph)
        
        # Identify critical nodes (high centrality)
        critical_nodes = []
        for node in self.graph.nodes():
            score = 0
            for metric in ['degree', 'betweenness', 'pagerank']:
                if metric in self.centrality_scores:
                    score += self.centrality_scores[metric].get(node, 0)
            
            if score > 0.5:  # Threshold for critical nodes
                critical_nodes.append(node)
        
        self.centrality_scores['critical_nodes'] = critical_nodes[:20]  # Top 20
    
    def _detect_communities(self):
        """Detect communities in the graph"""
        if self.graph.number_of_nodes() < 2:
            return
        
        # Convert to undirected for community detection
        undirected_graph = self.graph.to_undirected()
        
        try:
            # Use Louvain community detection
            import community as community_louvain
            partition = community_louvain.best_partition(undirected_graph)
            
            # Group nodes by community
            communities = defaultdict(list)
            for node, comm_id in partition.items():
                communities[comm_id].append(node)
            
            self.communities = list(communities.values())
            
        except ImportError:
            # Fallback to connected components
            self.communities = list(nx.weakly_connected_components(self.graph))
        
        logger.info(f"Detected {len(self.communities)} communities")
    
    def _find_critical_paths(self) -> List[Dict]:
        """Find critical paths in the infrastructure"""
        critical_paths = []
        
        # Find paths between critical nodes
        critical_nodes = self.centrality_scores.get('critical_nodes', [])
        
        for i, source in enumerate(critical_nodes[:5]):  # Limit for performance
            for target in critical_nodes[i+1:6]:
                try:
                    # Find shortest path
                    path = nx.shortest_path(self.graph, source, target)
                    
                    if len(path) > 2:  # Non-trivial paths
                        critical_paths.append({
                            'source': source,
                            'target': target,
                            'path': path,
                            'length': len(path),
                            'type': 'shortest'
                        })
                except nx.NetworkXNoPath:
                    continue
        
        # Find single points of failure
        articulation_points = []
        if self.graph.number_of_nodes() < 1000:  # Expensive
            undirected = self.graph.to_undirected()
            articulation_points = list(nx.articulation_points(undirected))
        
        if articulation_points:
            critical_paths.append({
                'type': 'single_points_of_failure',
                'nodes': articulation_points[:10],
                'impact': 'Removing these nodes would disconnect the graph'
            })
        
        return critical_paths
    
    async def _generate_embeddings(self):
        """Generate node and edge embeddings"""
        if self.graph.number_of_nodes() == 0:
            return
        
        # Simple node2vec style embeddings
        try:
            from node2vec import Node2Vec
            
            # Generate walks
            node2vec = Node2Vec(
                self.graph,
                dimensions=64,
                walk_length=10,
                num_walks=80,
                workers=4
            )
            
            # Learn embeddings
            model = node2vec.fit(window=10, min_count=1, batch_words=4)
            
            # Store embeddings
            for node in self.graph.nodes():
                if node in model.wv:
                    self.node_embeddings[node] = model.wv[node]
                    
        except ImportError:
            # Fallback to simple feature-based embeddings
            for node in self.graph.nodes():
                features = []
                node_data = self.graph.nodes[node]
                
                # Encode categorical features
                features.append(hash(node_data.get('type', '')) % 1000 / 1000)
                features.append(hash(node_data.get('environment', '')) % 1000 / 1000)
                features.append(hash(node_data.get('datacenter', '')) % 1000 / 1000)
                features.append(node_data.get('confidence', 0))
                features.append(self.graph.degree(node) / 100)
                
                # Pad to fixed size
                while len(features) < 64:
                    features.append(0)
                
                self.node_embeddings[node] = np.array(features[:64])
    
    def _build_hierarchy(self) -> Dict:
        """Build hierarchical structure of infrastructure"""
        hierarchy = {
            'environments': defaultdict(lambda: {
                'datacenters': defaultdict(lambda: {
                    'applications': defaultdict(list)
                })
            })
        }
        
        for node in self.graph.nodes():
            node_data = self.graph.nodes[node]
            env = node_data.get('environment', 'unknown')
            dc = node_data.get('datacenter', 'unknown')
            app = node_data.get('application', 'unknown')
            
            hierarchy['environments'][env]['datacenters'][dc]['applications'][app].append(node)
        
        # Convert defaultdicts to regular dicts for JSON serialization
        return json.loads(json.dumps(hierarchy, default=str))
    
    def _export_nodes(self) -> List[Dict]:
        """Export nodes with attributes"""
        nodes = []
        
        for node_id in self.graph.nodes():
            node_data = dict(self.graph.nodes[node_id])
            node_data['id'] = node_id
            node_data['degree'] = self.graph.degree(node_id)
            
            # Add centrality scores
            for metric in ['degree', 'betweenness', 'closeness', 'pagerank']:
                if metric in self.centrality_scores:
                    node_data[f'centrality_{metric}'] = self.centrality_scores[metric].get(node_id, 0)
            
            # Add embedding if available
            if node_id in self.node_embeddings:
                node_data['has_embedding'] = True
                # Don't include full embedding in export (too large)
            
            nodes.append(node_data)
        
        return nodes
    
    def _export_edges(self) -> List[Dict]:
        """Export edges with attributes"""
        edges = []
        
        for source, target, data in self.graph.edges(data=True):
            edge_data = dict(data)
            edge_data['source'] = source
            edge_data['target'] = target
            edges.append(edge_data)
        
        return edges
    
    def _export_communities(self) -> List[Dict]:
        """Export detected communities"""
        communities = []
        
        for i, community_nodes in enumerate(self.communities):
            # Calculate community statistics
            subgraph = self.graph.subgraph(community_nodes)
            
            community_data = {
                'id': i,
                'size': len(community_nodes),
                'nodes': list(community_nodes)[:20],  # Sample for large communities
                'density': nx.density(subgraph) if len(community_nodes) > 1 else 0,
                'internal_edges': subgraph.number_of_edges()
            }
            
            # Determine community type based on common attributes
            node_types = Counter()
            environments = Counter()
            applications = Counter()
            
            for node in community_nodes[:50]:  # Sample
                if node in self.graph:
                    node_data = self.graph.nodes[node]
                    node_types[node_data.get('type', 'unknown')] += 1
                    environments[node_data.get('environment', 'unknown')] += 1
                    applications[node_data.get('application', 'unknown')] += 1
            
            # Most common attributes
            if node_types:
                community_data['primary_type'] = node_types.most_common(1)[0][0]
            if environments:
                community_data['primary_environment'] = environments.most_common(1)[0][0]
            if applications:
                community_data['primary_application'] = applications.most_common(1)[0][0]
            
            communities.append(community_data)
        
        return communities
    
    def find_dependencies(self, node_id: str, depth: int = 2) -> Dict:
        """Find dependencies for a specific node"""
        if node_id not in self.graph:
            return {}
        
        dependencies = {
            'upstream': [],
            'downstream': []
        }
        
        # Find upstream dependencies (nodes this node depends on)
        for predecessor in self.graph.predecessors(node_id):
            edge_data = self.graph.edges[predecessor, node_id]
            dependencies['upstream'].append({
                'node': predecessor,
                'relationship': edge_data.get('type', 'unknown'),
                'confidence': edge_data.get('confidence', 0)
            })
        
        # Find downstream dependencies (nodes that depend on this node)
        for successor in self.graph.successors(node_id):
            edge_data = self.graph.edges[node_id, successor]
            dependencies['downstream'].append({
                'node': successor,
                'relationship': edge_data.get('type', 'unknown'),
                'confidence': edge_data.get('confidence', 0)
            })
        
        # Find multi-hop dependencies if requested
        if depth > 1:
            dependencies['multi_hop_upstream'] = []
            dependencies['multi_hop_downstream'] = []
            
            # BFS for multi-hop
            visited = {node_id}
            queue = [(node_id, 0)]
            
            while queue:
                current, current_depth = queue.pop(0)
                
                if current_depth < depth:
                    # Upstream
                    for pred in self.graph.predecessors(current):
                        if pred not in visited:
                            visited.add(pred)
                            queue.append((pred, current_depth + 1))
                            if current != node_id:
                                dependencies['multi_hop_upstream'].append({
                                    'node': pred,
                                    'hops': current_depth + 1
                                })
        
        return dependencies
    
    def get_impact_analysis(self, node_id: str) -> Dict:
        """Analyze impact of node failure"""
        if node_id not in self.graph:
            return {}
        
        impact = {
            'direct_impact': [],
            'total_affected': 0,
            'critical_paths_affected': 0,
            'communities_affected': []
        }
        
        # Find directly connected nodes
        impact['direct_impact'] = list(self.graph.neighbors(node_id))
        
        # Find all reachable nodes
        reachable = nx.descendants(self.graph, node_id)
        impact['total_affected'] = len(reachable)
        
        # Check critical paths
        for path_info in self.centrality_scores.get('critical_paths', []):
            if node_id in path_info.get('path', []):
                impact['critical_paths_affected'] += 1
        
        # Check communities
        for i, community in enumerate(self.communities):
            if node_id in community:
                impact['communities_affected'].append(i)
        
        return impact
    
    def find_similar_nodes(self, node_id: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Find similar nodes based on attributes and embeddings"""
        if node_id not in self.graph:
            return []
        
        similar_nodes = []
        source_data = self.graph.nodes[node_id]
        source_embedding = self.node_embeddings.get(node_id)
        
        for other_node in self.graph.nodes():
            if other_node == node_id:
                continue
            
            similarity_score = 0.0
            other_data = self.graph.nodes[other_node]
            
            # Attribute similarity
            if source_data.get('type') == other_data.get('type'):
                similarity_score += 0.3
            if source_data.get('environment') == other_data.get('environment'):
                similarity_score += 0.2
            if source_data.get('datacenter') == other_data.get('datacenter'):
                similarity_score += 0.1
            if source_data.get('application') == other_data.get('application'):
                similarity_score += 0.2
            
            # Embedding similarity
            if source_embedding is not None and other_node in self.node_embeddings:
                other_embedding = self.node_embeddings[other_node]
                # Cosine similarity
                cosine_sim = np.dot(source_embedding, other_embedding) / (
                    np.linalg.norm(source_embedding) * np.linalg.norm(other_embedding)
                )
                similarity_score += cosine_sim * 0.2
            
            similar_nodes.append((other_node, similarity_score))
        
        # Sort by similarity and return top k
        similar_nodes.sort(key=lambda x: x[1], reverse=True)
        return similar_nodes[:top_k]
    
    def get_subgraph(self, nodes: List[str], include_edges: bool = True) -> nx.MultiDiGraph:
        """Get subgraph containing specified nodes"""
        if include_edges:
            return self.graph.subgraph(nodes).copy()
        else:
            # Create subgraph without edges between nodes
            subgraph = nx.MultiDiGraph()
            for node in nodes:
                if node in self.graph:
                    subgraph.add_node(node, **self.graph.nodes[node])
            return subgraph
    
    def find_shortest_path(self, source: str, target: str) -> Optional[List[str]]:
        """Find shortest path between two nodes"""
        try:
            return nx.shortest_path(self.graph, source, target)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None
    
    def get_node_statistics(self) -> Dict:
        """Get overall statistics about nodes in the graph"""
        stats = {
            'total_nodes': self.graph.number_of_nodes(),
            'total_edges': self.graph.number_of_edges(),
            'node_types': Counter(),
            'environments': Counter(),
            'datacenters': Counter(),
            'applications': Counter(),
            'criticality_levels': Counter(),
            'average_degree': 0,
            'max_degree': 0,
            'min_degree': 0,
            'isolated_nodes': 0
        }
        
        degrees = []
        for node in self.graph.nodes():
            node_data = self.graph.nodes[node]
            degree = self.graph.degree(node)
            degrees.append(degree)
            
            # Count attributes
            stats['node_types'][node_data.get('type', 'unknown')] += 1
            stats['environments'][node_data.get('environment', 'unknown')] += 1
            stats['datacenters'][node_data.get('datacenter', 'unknown')] += 1
            stats['applications'][node_data.get('application', 'unknown')] += 1
            stats['criticality_levels'][node_data.get('criticality', 'medium')] += 1
            
            if degree == 0:
                stats['isolated_nodes'] += 1
        
        if degrees:
            stats['average_degree'] = sum(degrees) / len(degrees)
            stats['max_degree'] = max(degrees)
            stats['min_degree'] = min(degrees)
        
        # Convert Counters to dicts for JSON serialization
        stats['node_types'] = dict(stats['node_types'])
        stats['environments'] = dict(stats['environments'])
        stats['datacenters'] = dict(stats['datacenters'])
        stats['applications'] = dict(stats['applications'])
        stats['criticality_levels'] = dict(stats['criticality_levels'])
        
        return stats
    
    def export_to_graphml(self, filepath: str):
        """Export graph to GraphML format for visualization"""
        nx.write_graphml(self.graph, filepath)
        logger.info(f"Graph exported to {filepath}")
    
    def export_to_cytoscape(self) -> Dict:
        """Export graph in Cytoscape.js format"""
        elements = {
            'nodes': [],
            'edges': []
        }
        
        # Export nodes
        for node_id in self.graph.nodes():
            node_data = self.graph.nodes[node_id]
            elements['nodes'].append({
                'data': {
                    'id': node_id,
                    'label': node_id,
                    **{k: v for k, v in node_data.items() if not isinstance(v, (list, dict))}
                }
            })
        
        # Export edges
        for source, target, data in self.graph.edges(data=True):
            elements['edges'].append({
                'data': {
                    'id': f"{source}-{target}",
                    'source': source,
                    'target': target,
                    'label': data.get('type', 'related'),
                    **{k: v for k, v in data.items() if not isinstance(v, (list, dict))}
                }
            })
        
        return elements