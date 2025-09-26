# processors/entity_resolver.py
"""
Entity Resolver - Deduplicates and resolves entities across all tables
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Set, Tuple, Optional
import re
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

# Try to import faiss, but make it optional
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.debug("FAISS not available, using fallback similarity search")

class TextEncoder(nn.Module):
    """Simple text encoder to replace SentenceTransformer"""
    def __init__(self, embed_dim: int = 384, device: str = 'cpu'):
        super().__init__()
        self.device = device if torch.cuda.is_available() or device == 'cpu' else 'cpu'
        
        # Character-level embedding
        self.char_embed = nn.Embedding(256, 64)
        
        # Convolutional layers
        self.conv1 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(128, 256, kernel_size=5, padding=2)
        
        # Pooling and projection
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.projection = nn.Linear(256, embed_dim)
        
        self.to(self.device)
    
    def encode(self, text: str) -> np.ndarray:
        """Encode text to embedding"""
        # Convert to character indices
        chars = [ord(c) % 256 for c in text[:512]]
        if not chars:
            return np.zeros(384, dtype=np.float32)
        
        char_tensor = torch.tensor(chars, device=self.device).unsqueeze(0)
        
        with torch.no_grad():
            # Embed and convolve
            x = self.char_embed(char_tensor).transpose(1, 2)
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            
            # Pool and project
            x = self.pool(x).squeeze(-1)
            embedding = self.projection(x)
        
        return embedding.cpu().numpy().flatten()

class EntityResolver:
    def __init__(self, config: Dict = None, device: str = 'cpu'):
        if config is None:
            config = {
                'entity_resolution': {
                    'comparison_threshold': 0.7,
                    'confidence_levels': [0.8, 0.9, 0.95]
                }
            }
        
        self.config = config
        self.device = device if torch.cuda.is_available() or device == 'cpu' else 'cpu'
        
        # Use simple text encoder
        self.encoder = TextEncoder(device=self.device)
        
        # Get config values with defaults
        entity_config = config.get('entity_resolution', {})
        self.blocking_threshold = entity_config.get('comparison_threshold', 0.7)
        self.confidence_levels = entity_config.get('confidence_levels', [0.8, 0.9, 0.95])
        
        self.lsh_index = None
        self.entity_embeddings = {}
        self.blocking_keys = defaultdict(set)
        
        self.similarity_model = SimilarityModel(self.device)
    
    async def resolve(self, hosts: Dict, metadata: Dict = None) -> Dict:
        """Resolve and deduplicate entities"""
        if metadata is None:
            metadata = {}
            
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
        embeddings = []
        entity_list = []
        
        for hostname, data in hosts.items():
            # Create text representation
            raw_forms = data.get('raw_forms', [])
            if isinstance(raw_forms, set):
                raw_forms = list(raw_forms)
            text = f"{hostname} " + " ".join(str(f) for f in raw_forms[:10])
            
            # Get embedding using our encoder
            embedding = self.encoder.encode(text)
            
            embeddings.append(embedding)
            entity_list.append(hostname)
            self.entity_embeddings[hostname] = embedding
            
            # Generate blocking keys
            for blocking_key in self._generate_blocking_keys(hostname):
                self.blocking_keys[blocking_key].add(hostname)
        
        # Build LSH index if FAISS available
        if FAISS_AVAILABLE and embeddings:
            embeddings_matrix = np.array(embeddings).astype('float32')
            
            dim = embeddings_matrix.shape[1]
            # Use simple index for small datasets
            if len(embeddings) < 10000:
                self.lsh_index = faiss.IndexFlatL2(dim)
            else:
                # Use LSH for larger datasets
                self.lsh_index = faiss.IndexLSHF(dim, 256)
            self.lsh_index.add(embeddings_matrix)
    
    def _generate_blocking_keys(self, hostname: str) -> Set[str]:
        """Generate blocking keys for candidate generation"""
        keys = set()
        
        hostname_lower = hostname.lower()
        
        # First 3 characters
        if len(hostname_lower) >= 3:
            keys.add(hostname_lower[:3])
        
        # Split by delimiters
        parts = re.split(r'[\.\-_]', hostname_lower)
        for part in parts:
            if len(part) >= 3:
                keys.add(part[:3])
        
        # IP prefix
        ip_match = re.match(r'^(\d+\.\d+)', hostname)
        if ip_match:
            keys.add(ip_match.group(1))
        
        # Domain
        domain_match = re.search(r'([a-z0-9\-]+)\.[a-z]{2,}$', hostname_lower)
        if domain_match:
            keys.add(domain_match.group(1))
        
        return keys
    
    def _generate_candidates(self, hosts: Dict) -> List[Tuple[str, str]]:
        """Generate candidate pairs for comparison"""
        candidates = set()
        
        # Blocking-based candidates
        for key, entities in self.blocking_keys.items():
            if len(entities) > 1 and len(entities) < 100:  # Avoid very large blocks
                entity_list = list(entities)
                for i in range(len(entity_list)):
                    for j in range(i + 1, min(i + 10, len(entity_list))):  # Limit comparisons
                        candidates.add(tuple(sorted([entity_list[i], entity_list[j]])))
        
        # LSH-based candidates (if available and not too many hosts)
        if FAISS_AVAILABLE and self.lsh_index and len(hosts) < 10000:
            host_list = list(hosts.keys())
            for i, hostname in enumerate(host_list[:1000]):  # Limit for performance
                if hostname in self.entity_embeddings:
                    query = self.entity_embeddings[hostname].reshape(1, -1)
                    # Search for k nearest neighbors
                    k = min(10, len(hosts))
                    D, I = self.lsh_index.search(query, k)
                    
                    for idx in I[0]:
                        if 0 <= idx < len(host_list) and idx != i:
                            candidate = host_list[idx]
                            if candidate != hostname:
                                candidates.add(tuple(sorted([hostname, candidate])))
        
        return list(candidates)
    
    async def _calculate_similarity(self, entity1_data: Dict, entity2_data: Dict) -> float:
        """Calculate similarity between two entities"""
        scores = []
        
        # Name similarity
        name1 = str(entity1_data.get('hostname', ''))
        name2 = str(entity2_data.get('hostname', ''))
        name_sim = self._string_similarity(name1, name2)
        scores.append(name_sim * 0.4)
        
        # Raw forms similarity
        raw_forms1 = entity1_data.get('raw_forms', [])
        raw_forms2 = entity2_data.get('raw_forms', [])
        
        # Convert to sets if needed
        if isinstance(raw_forms1, list):
            raw_forms1 = set(raw_forms1)
        if isinstance(raw_forms2, list):
            raw_forms2 = set(raw_forms2)
            
        if raw_forms1 and raw_forms2:
            jaccard = len(raw_forms1 & raw_forms2) / len(raw_forms1 | raw_forms2)
            scores.append(jaccard * 0.3)
        
        # Attribute similarity
        attrs1 = entity1_data.get('attributes', {})
        attrs2 = entity2_data.get('attributes', {})
        
        common_attrs = set(attrs1.keys()) & set(attrs2.keys())
        if common_attrs:
            attr_similarities = []
            for attr in list(common_attrs)[:20]:  # Limit for performance
                val1 = attrs1[attr]
                val2 = attrs2[attr]
                
                if isinstance(val1, list) and val1:
                    val1 = val1[0]
                if isinstance(val2, list) and val2:
                    val2 = val2[0]
                
                if val1 and val2:
                    attr_sim = self._string_similarity(str(val1), str(val2))
                    attr_similarities.append(attr_sim)
            
            if attr_similarities:
                scores.append(np.mean(attr_similarities) * 0.3)
        
        return sum(scores)
    
    def _string_similarity(self, s1: str, s2: str) -> float:
        """Calculate string similarity"""
        s1, s2 = s1.lower(), s2.lower()
        
        if s1 == s2:
            return 1.0
        
        # Use simple character-based similarity
        from difflib import SequenceMatcher
        return SequenceMatcher(None, s1, s2).ratio()
    
    def _select_master(self, entities: List[str]) -> str:
        """Select the master entity from candidates"""
        scores = {}
        
        for entity in entities:
            score = 0
            
            # Prefer FQDNs
            if '.' in entity:
                score += 2
            
            # Prefer non-numeric
            if not any(c.isdigit() for c in entity):
                score += 1
            
            # Length bonus
            score += len(entity)
            
            scores[entity] = score
        
        return max(scores, key=scores.get)
    
    def _merge_entities(self, entity1: Dict, entity2: Dict) -> Dict:
        """Merge two entities into one"""
        merged = {
            'raw_forms': set(),
            'occurrences': [],
            'attributes': defaultdict(list)
        }
        
        # Merge raw forms
        raw1 = entity1.get('raw_forms', [])
        raw2 = entity2.get('raw_forms', [])
        
        if isinstance(raw1, set):
            merged['raw_forms'].update(raw1)
        else:
            merged['raw_forms'].update(raw1)
            
        if isinstance(raw2, set):
            merged['raw_forms'].update(raw2)
        else:
            merged['raw_forms'].update(raw2)
        
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
        
        # Deduplicate attribute values
        for attr in merged['attributes']:
            values = merged['attributes'][attr]
            unique_values = []
            seen = set()
            
            for val in values:
                val_str = str(val).lower()
                if val_str not in seen:
                    unique_values.append(val)
                    seen.add(val_str)
            
            merged['attributes'][attr] = unique_values[:10]  # Limit to 10 values
        
        # Convert raw_forms back to list for serialization
        merged['raw_forms'] = list(merged['raw_forms'])
        
        return merged

class SimilarityModel(nn.Module):
    """Neural network model for similarity scoring"""
    def __init__(self, device: str = 'cpu'):
        super().__init__()
        
        self.device = device if torch.cuda.is_available() or device == 'cpu' else 'cpu'
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
        
        self.to(self.device)
    
    def forward(self, emb1, emb2):
        combined = torch.cat([emb1, emb2], dim=-1)
        features = self.encoder(combined)
        similarity = self.classifier(features)
        return similarity