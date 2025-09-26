import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Set, Tuple
import networkx as nx
from collections import defaultdict
import re

class LocalSimilarityEncoder(nn.Module):
    """Local similarity encoder without external dependencies"""
    def __init__(self, input_dim: int = 384, hidden_dim: int = 256, device: str = 'mps'):
        super().__init__()
        self.device = device
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 64)
        )
        
        self.to(device)
    
    def forward(self, x):
        return self.encoder(x)
    
    def encode_text(self, text: str) -> np.ndarray:
        """Simple text encoding using character-level features"""
        features = []
        
        # Length features
        features.append(len(text))
        features.append(len(text.split()))
        
        # Character type ratios
        if text:
            features.append(sum(c.isalpha() for c in text) / len(text))
            features.append(sum(c.isdigit() for c in text) / len(text))
            features.append(sum(c.isspace() for c in text) / len(text))
            features.append(sum(c in '.-_' for c in text) / len(text))
        else:
            features.extend([0] * 4)
        
        # N-gram features
        bigrams = [text[i:i+2] for i in range(max(0, len(text)-1))]
        trigrams = [text[i:i+3] for i in range(max(0, len(text)-2))]
        
        # Common bigram/trigram counts
        common_bigrams = ['th', 'he', 'in', 'er', 'an', 're', 'ed', 'on', 'es', 'st']
        for bg in common_bigrams:
            features.append(bigrams.count(bg) / max(len(bigrams), 1))
        
        # Hash-based features for capturing unique patterns
        text_hash = hash(text.lower())
        for i in range(20):
            features.append((text_hash >> i) & 1)
        
        # Word-level features
        words = text.lower().split()
        if words:
            # Average word length
            features.append(np.mean([len(w) for w in words]))
            # Number of unique words
            features.append(len(set(words)) / len(words))
        else:
            features.extend([0, 0])
        
        # Pad to fixed size
        while len(features) < self.encoder[0].in_features:
            features.append(0)
        
        features = np.array(features[:self.encoder[0].in_features], dtype=np.float32)
        
        with torch.no_grad():
            tensor = torch.tensor(features, device=self.device).unsqueeze(0)
            encoded = self.forward(tensor)
        
        return encoded.cpu().numpy().flatten()

class SimpleLSH:
    """Simple Locality Sensitive Hashing implementation"""
    def __init__(self, num_bits: int = 256, num_tables: int = 10):
        self.num_bits = num_bits
        self.num_tables = num_tables
        self.tables = defaultdict(lambda: defaultdict(list))
        self.random_vectors = []
        
        # Initialize random projection vectors
        for i in range(num_tables):
            random_vec = np.random.randn(num_bits, 64)
            random_vec /= np.linalg.norm(random_vec, axis=1, keepdims=True)
            self.random_vectors.append(random_vec)
    
    def _hash(self, vector: np.ndarray, table_idx: int) -> str:
        """Generate hash for vector"""
        projections = np.dot(self.random_vectors[table_idx], vector)
        hash_bits = (projections > 0).astype(int)
        return ''.join(map(str, hash_bits))
    
    def add(self, key: str, vector: np.ndarray):
        """Add vector to LSH tables"""
        for i in range(self.num_tables):
            hash_key = self._hash(vector, i)
            self.tables[i][hash_key].append(key)
    
    def query(self, vector: np.ndarray, k: int = 10) -> List[str]:
        """Query similar items"""
        candidates = set()
        
        for i in range(self.num_tables):
            hash_key = self._hash(vector, i)
            if hash_key in self.tables[i]:
                candidates.update(self.tables[i][hash_key])
        
        return list(candidates)[:k]

class EntityResolver:
    def __init__(self, config: Dict, device: str = 'mps'):
        self.config = config
        self.device = device
        
        # Local similarity encoder
        self.encoder = LocalSimilarityEncoder(device=device)
        
        # LSH for blocking
        self.lsh = SimpleLSH(num_bits=256, num_tables=10)
        
        self.blocking_threshold = config['entity_resolution']['comparison_threshold']
        self.confidence_levels = config['entity_resolution']['confidence_levels']
        
        self.entity_embeddings = {}
        self.blocking_keys = defaultdict(set)
        
        # Similarity model
        self.similarity_model = SimilarityModel(device)
    
    async def resolve(self, hosts: Dict, metadata: Dict) -> Dict:
        """Resolve entities"""
        # Build blocking index
        self._build_blocking_index(hosts)
        
        # Generate candidate pairs
        candidate_pairs = self._generate_candidates(hosts)
        
        # Resolve entities
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
        
        # Add unprocessed entities
        for entity in hosts:
            if entity not in processed:
                resolved_entities[entity] = hosts[entity]
        
        return resolved_entities
    
    def _build_blocking_index(self, hosts: Dict):
        """Build blocking index for efficient candidate generation"""
        for hostname, data in hosts.items():
            # Create text representation
            text = f"{hostname} " + " ".join(str(f) for f in data.get('raw_forms', []))
            
            # Generate embedding
            embedding = self.encoder.encode_text(text)
            self.entity_embeddings[hostname] = embedding
            
            # Add to LSH
            self.lsh.add(hostname, embedding)
            
            # Generate blocking keys
            for blocking_key in self._generate_blocking_keys(hostname):
                self.blocking_keys[blocking_key].add(hostname)
    
    def _generate_blocking_keys(self, hostname: str) -> Set[str]:
        """Generate blocking keys for hostname"""
        keys = set()
        
        hostname_lower = hostname.lower()
        
        # Prefix blocking
        keys.add(hostname_lower[:3])
        
        # Token blocking
        parts = re.split(r'[\.\-_]', hostname_lower)
        for part in parts:
            if len(part) >= 3:
                keys.add(part[:3])
        
        # IP prefix blocking
        ip_match = re.match(r'^(\d+\.\d+)', hostname)
        if ip_match:
            keys.add(ip_match.group(1))
        
        # Domain blocking
        domain_match = re.search(r'([a-z0-9\-]+)\.[a-z]{2,}$', hostname_lower)
        if domain_match:
            keys.add(domain_match.group(1))
        
        return keys
    
    def _generate_candidates(self, hosts: Dict) -> List[Tuple[str, str]]:
        """Generate candidate pairs for comparison"""
        candidates = set()
        
        # Blocking-based candidates
        for key, entities in self.blocking_keys.items():
            if len(entities) > 1:
                entity_list = list(entities)
                for i in range(len(entity_list)):
                    for j in range(i + 1, min(i + 10, len(entity_list))):
                        candidates.add(tuple(sorted([entity_list[i], entity_list[j]])))
        
        # LSH-based candidates
        for hostname in list(hosts.keys())[:1000]:
            if hostname in self.entity_embeddings:
                similar = self.lsh.query(self.entity_embeddings[hostname], k=10)
                for candidate in similar:
                    if candidate != hostname:
                        candidates.add(tuple(sorted([hostname, candidate])))
        
        return list(candidates)
    
    async def _calculate_similarity(self, entity1_data: Dict, entity2_data: Dict) -> float:
        """Calculate similarity between two entities"""
        scores = []
        
        # Name similarity
        name_sim = self._string_similarity(
            str(entity1_data.get('hostname', '')),
            str(entity2_data.get('hostname', ''))
        )
        scores.append(name_sim * 0.4)
        
        # Raw forms similarity
        raw_forms1 = set(str(f) for f in entity1_data.get('raw_forms', []))
        raw_forms2 = set(str(f) for f in entity2_data.get('raw_forms', []))
        
        if raw_forms1 and raw_forms2:
            jaccard = len(raw_forms1 & raw_forms2) / len(raw_forms1 | raw_forms2)
            scores.append(jaccard * 0.3)
        
        # Attribute similarity
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
            
            if attr_similarities:
                scores.append(np.mean(attr_similarities) * 0.3)
        
        return sum(scores)
    
    def _string_similarity(self, s1: str, s2: str) -> float:
        """Calculate string similarity using multiple methods"""
        s1, s2 = s1.lower(), s2.lower()
        
        if s1 == s2:
            return 1.0
        
        # Jaccard similarity on character bigrams
        def get_bigrams(s):
            return set(s[i:i+2] for i in range(len(s)-1))
        
        bigrams1 = get_bigrams(s1)
        bigrams2 = get_bigrams(s2)
        
        if bigrams1 or bigrams2:
            jaccard = len(bigrams1 & bigrams2) / len(bigrams1 | bigrams2) if (bigrams1 | bigrams2) else 0
        else:
            jaccard = 0
        
        # Levenshtein distance (simple implementation)
        def levenshtein(s1, s2):
            if len(s1) < len(s2):
                s1, s2 = s2, s1
            
            if len(s2) == 0:
                return 0.0
            
            distances = range(len(s2) + 1)
            
            for i, c1 in enumerate(s1):
                new_distances = [i + 1]
                for j, c2 in enumerate(s2):
                    if c1 == c2:
                        new_distances.append(distances[j])
                    else:
                        new_distances.append(1 + min(distances[j], distances[j + 1], new_distances[-1]))
                distances = new_distances
            
            return 1.0 - (distances[-1] / max(len(s1), len(s2)))
        
        lev_sim = levenshtein(s1, s2)
        
        # Weighted average
        return (jaccard * 0.5 + lev_sim * 0.5)
    
    def _select_master(self, entities: List[str]) -> str:
        """Select master entity from candidates"""
        scores = {}
        
        for entity in entities:
            score = 0
            
            # Prefer FQDNs
            if '.' in entity:
                score += 2
            
            # Prefer non-numeric
            if not any(c.isdigit() for c in entity):
                score += 1
            
            # Prefer longer names
            score += len(entity) * 0.1
            
            scores[entity] = score
        
        return max(scores, key=scores.get)
    
    def _merge_entities(self, entity1: Dict, entity2: Dict) -> Dict:
        """Merge two entities"""
        merged = {
            'raw_forms': set(),
            'occurrences': [],
            'attributes': defaultdict(list)
        }
        
        # Merge raw forms
        merged['raw_forms'].update(entity1.get('raw_forms', []))
        merged['raw_forms'].update(entity2.get('raw_forms', []))
        
        # Merge occurrences
        merged['occurrences'].extend(entity1.get('occurrences', []))
        merged['occurrences'].extend(entity2.get('occurrences', []))
        
        # Merge attributes
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
        
        # Deduplicate attributes
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
    """Neural similarity model"""
    def __init__(self, device: str = 'mps'):
        super().__init__()
        
        self.embedding_dim = 64
        
        self.encoder = nn.Sequential(
            nn.Linear(self.embedding_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
        self.to(device)
    
    def forward(self, emb1, emb2):
        combined = torch.cat([emb1, emb2], dim=-1)
        features = self.encoder(combined)
        similarity = self.classifier(features)
        return similarity