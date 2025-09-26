import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Set, Tuple
import faiss
from sentence_transformers import SentenceTransformer
import networkx as nx
from collections import defaultdict
import re

class EntityResolver:
    def __init__(self, config: Dict, device: str = 'mps'):
        self.config = config
        self.device = device
        
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.encoder.to(device)
        
        self.blocking_threshold = config['entity_resolution']['comparison_threshold']
        self.confidence_levels = config['entity_resolution']['confidence_levels']
        
        self.lsh_index = None
        self.entity_embeddings = {}
        self.blocking_keys = defaultdict(set)
        
        self.similarity_model = SimilarityModel(device)
    
    async def resolve(self, hosts: Dict, metadata: Dict) -> Dict:
        self._build_blocking_index(hosts)
        
        candidate_pairs = self._generate_candidates(hosts)
        
        resolved_entities = {}
        processed = set()
        
        for entity1, entity2 in candidate_pairs:
            if entity1 in processed or entity2 in processed:
                continue
            
            similarity = await self._calculate_similarity(
                hosts[entity1],
                hosts[entity2]
            )
            
            if similarity > self.confidence_levels[0]:
                master = self._select_master([entity1, entity2])
                resolved_entities[master] = self._merge_entities(
                    hosts[entity1],
                    hosts[entity2]
                )
                processed.update([entity1, entity2])
        
        for entity in hosts:
            if entity not in processed:
                resolved_entities[entity] = hosts[entity]
        
        return resolved_entities
    
    def _build_blocking_index(self, hosts: Dict):
        embeddings = []
        entity_list = []
        
        for hostname, data in hosts.items():
            text = f"{hostname} " + " ".join(data.get('raw_forms', []))
            embedding = self.encoder.encode(text, convert_to_tensor=True, device=self.device)
            embedding_np = embedding.cpu().numpy()
            
            embeddings.append(embedding_np)
            entity_list.append(hostname)
            self.entity_embeddings[hostname] = embedding_np
            
            for blocking_key in self._generate_blocking_keys(hostname):
                self.blocking_keys[blocking_key].add(hostname)
        
        if embeddings:
            embeddings_matrix = np.array(embeddings).astype('float32')
            
            dim = embeddings_matrix.shape[1]
            self.lsh_index = faiss.IndexLSHF(dim, 256)
            self.lsh_index.add(embeddings_matrix)
    
    def _generate_blocking_keys(self, hostname: str) -> Set[str]:
        keys = set()
        
        hostname_lower = hostname.lower()
        keys.add(hostname_lower[:3])
        
        parts = re.split(r'[\.\-_]', hostname_lower)
        for part in parts:
            if len(part) >= 3:
                keys.add(part[:3])
        
        ip_match = re.match(r'^(\d+\.\d+)', hostname)
        if ip_match:
            keys.add(ip_match.group(1))
        
        domain_match = re.search(r'([a-z0-9\-]+)\.[a-z]{2,}$', hostname_lower)
        if domain_match:
            keys.add(domain_match.group(1))
        
        return keys
    
    def _generate_candidates(self, hosts: Dict) -> List[Tuple[str, str]]:
        candidates = set()
        
        for key, entities in self.blocking_keys.items():
            if len(entities) > 1:
                entity_list = list(entities)
                for i in range(len(entity_list)):
                    for j in range(i + 1, len(entity_list)):
                        candidates.add(tuple(sorted([entity_list[i], entity_list[j]])))
        
        if self.lsh_index and len(hosts) < 10000:
            for hostname in list(hosts.keys())[:1000]:
                if hostname in self.entity_embeddings:
                    query = self.entity_embeddings[hostname].reshape(1, -1)
                    D, I = self.lsh_index.search(query, 10)
                    
                    for idx in I[0]:
                        if idx >= 0 and idx < len(hosts):
                            candidate = list(hosts.keys())[idx]
                            if candidate != hostname:
                                candidates.add(tuple(sorted([hostname, candidate])))
        
        return list(candidates)
    
    async def _calculate_similarity(self, entity1_data: Dict, entity2_data: Dict) -> float:
        scores = []
        
        name_sim = self._string_similarity(
            str(entity1_data.get('hostname', '')),
            str(entity2_data.get('hostname', ''))
        )
        scores.append(name_sim * 0.4)
        
        raw_forms1 = set(entity1_data.get('raw_forms', []))
        raw_forms2 = set(entity2_data.get('raw_forms', []))
        if raw_forms1 and raw_forms2:
            jaccard = len(raw_forms1 & raw_forms2) / len(raw_forms1 | raw_forms2)
            scores.append(jaccard * 0.3)
        
        attrs1 = entity1_data.get('attributes', {})
        attrs2 = entity2_data.get('attributes', {})
        
        common_attrs = set(attrs1.keys()) & set(attrs2.keys())
        if common_attrs:
            attr_similarities = []
            for attr in common_attrs:
                val1 = attrs1[attr]
                val2 = attrs2[attr]
                
                if isinstance(val1, list):
                    val1 = val1[0] if val1 else ''
                if isinstance(val2, list):
                    val2 = val2[0] if val2 else ''
                
                attr_sim = self._string_similarity(str(val1), str(val2))
                attr_similarities.append(attr_sim)
            
            scores.append(np.mean(attr_similarities) * 0.3)
        
        return sum(scores)
    
    def _string_similarity(self, s1: str, s2: str) -> float:
        s1, s2 = s1.lower(), s2.lower()
        
        if s1 == s2:
            return 1.0
        
        from difflib import SequenceMatcher
        ratio = SequenceMatcher(None, s1, s2).ratio()
        
        return ratio
    
    def select_master(self, entities: List[str]) -> str:
        scores = {}
        
        for entity in entities:
            score = 0
            
            if '.' in entity:
                score += 2
            
            if not any(c.isdigit() for c in entity):
                score += 1
            
            score += len(entity)
            
            scores[entity] = score
        
        return max(scores, key=scores.get)
    
    def _merge_entities(self, entity1: Dict, entity2: Dict) -> Dict:
        merged = {
            'raw_forms': set(),
            'occurrences': [],
            'attributes': defaultdict(list)
        }
        
        merged['raw_forms'].update(entity1.get('raw_forms', []))
        merged['raw_forms'].update(entity2.get('raw_forms', []))
        
        merged['occurrences'].extend(entity1.get('occurrences', []))
        merged['occurrences'].extend(entity2.get('occurrences', []))
        
        for attr, values in entity1.get('attributes', {}).items():
            if isinstance(values, list):
                merged['attributes'][attr].extend(values)
            else:
                merged['attributes'][attr].append(values)
        
        for attr, values in entity2.get('attributes', {}).items():
            if isinstance(values, list):
                merged['attributes'][attr].extend(values)
            else:
                merged['attributes'][attr].append(values)
        
        for attr in merged['attributes']:
            values = merged['attributes'][attr]
            unique_values = []
            seen = set()
            
            for val in values:
                val_str = str(val).lower()
                if val_str not in seen:
                    unique_values.append(val)
                    seen.add(val_str)
            
            merged['attributes'][attr] = unique_values[:10]
        
        return merged

class SimilarityModel(nn.Module):
    def __init__(self, device: str = 'mps'):
        super().__init__()
        
        self.embedding_dim = 384
        
        self.encoder = nn.Sequential(
            nn.Linear(self.embedding_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        self.to(device)
    
    def forward(self, emb1, emb2):
        combined = torch.cat([emb1, emb2], dim=-1)
        features = self.encoder(combined)
        similarity = self.classifier(features)
        return similarity