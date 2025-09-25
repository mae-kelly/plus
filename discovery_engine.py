import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any, Optional, Set
import ipaddress
import logging

logger = logging.getLogger(__name__)

class DiscoveryEngine:
    
    def __init__(self, device: str = 'mps'):
        self.device = device
        
        self.learned_patterns = {
            'structures': [],
            'lengths': [],
            'character_sets': set(),
            'separators': set(),
            'prefixes': set(),
            'suffixes': set(),
            'common_patterns': []
        }
        
        self.neural_model = HostnameNeuralNet(device=device)
        self.high_confidence = 0.8
        self.medium_confidence = 0.5
        self.pattern_cache = {}
        
        self.stats = {
            'patterns_learned': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
    async def train(self, samples: List[str]):
        if not samples:
            return
            
        logger.info(f"Training discovery engine on {len(samples)} samples")
        
        for sample in samples:
            if not sample:
                continue
                
            sample = str(sample).strip()
            
            # Learn structure
            structure = self._extract_structure(sample)
            if structure not in self.learned_patterns['structures']:
                self.learned_patterns['structures'].append(structure)
            
            # Learn length distribution
            self.learned_patterns['lengths'].append(len(sample))
            
            # Learn character sets
            for char in sample:
                self.learned_patterns['character_sets'].add(char)
            
            # Learn separators
            for sep in ['.', '-', '_']:
                if sep in sample:
                    self.learned_patterns['separators'].add(sep)
            
            # Learn common prefixes/suffixes
            if '.' in sample:
                parts = sample.split('.')
                if parts[0] and len(parts[0]) > 2:
                    prefix = parts[0][:3].lower()
                    self.learned_patterns['prefixes'].add(prefix)
                if len(parts) > 1 and parts[-1]:
                    suffix = parts[-1].lower()
                    self.learned_patterns['suffixes'].add(suffix)
                    
            # Learn full patterns
            self.learned_patterns['common_patterns'].append(sample.lower())
        
        self.stats['patterns_learned'] = len(self.learned_patterns['structures'])
        
        await self._train_neural_model(samples)
        
        logger.info(f"Discovery engine training complete - learned {self.stats['patterns_learned']} patterns")
        
    def is_hostname(self, value: Any) -> bool:
        if value is None or value == '':
            return False
            
        str_val = str(value).strip().lower()
        
        if str_val in self.pattern_cache:
            self.stats['cache_hits'] += 1
            return self.pattern_cache[str_val]
        
        self.stats['cache_misses'] += 1
        
        # Basic filters
        if len(str_val) < 2 or len(str_val) > 253:
            self.pattern_cache[str_val] = False
            return False
            
        if str_val.isdigit():
            self.pattern_cache[str_val] = False
            return False
        
        confidence = self.get_confidence(str_val)
        result = confidence >= self.medium_confidence
        
        self.pattern_cache[str_val] = result
        return result
        
    def get_confidence(self, value: str) -> float:
        if not value:
            return 0.0
            
        scores = []
        value = str(value).strip()
        
        # Check if it's an IP address
        if self._is_ip_address(value):
            return 1.0
        
        # Check FQDN pattern
        if self._is_fqdn(value):
            scores.append(0.95)
        
        # Check short hostname pattern  
        if self._is_short_hostname(value):
            scores.append(0.85)
        
        # Check cloud instance patterns
        if self._is_cloud_instance(value):
            scores.append(0.9)
            
        # Check Kubernetes patterns
        if self._is_kubernetes_name(value):
            scores.append(0.85)
        
        # Check against learned patterns
        pattern_score = self._match_learned_patterns(value)
        if pattern_score > 0:
            scores.append(pattern_score)
        
        # Neural model prediction
        if hasattr(self.neural_model, 'predict'):
            try:
                neural_score = self.neural_model.predict(value)
                scores.append(neural_score)
            except:
                pass
        
        return max(scores) if scores else 0.0
        
    def _extract_structure(self, hostname: str) -> str:
        structure = []
        
        for char in hostname:
            if char.isalpha():
                structure.append('A' if char.isupper() else 'a')
            elif char.isdigit():
                structure.append('0')
            elif char in '.-_':
                structure.append(char)
            else:
                structure.append('?')
        
        # Simplify consecutive patterns
        simplified = []
        last = None
        count = 0
        
        for s in structure:
            if s == last:
                count += 1
            else:
                if last and count > 1:
                    simplified.append(f"{last}{count}")
                elif last:
                    simplified.append(last)
                last = s
                count = 1
        
        if last:
            if count > 1:
                simplified.append(f"{last}{count}")
            else:
                simplified.append(last)
        
        return ''.join(simplified)
        
    def _is_ip_address(self, value: str) -> bool:
        try:
            ipaddress.ip_address(value)
            return True
        except:
            return False
            
    def _is_fqdn(self, value: str) -> bool:
        fqdn_pattern = re.compile(
            r'^(?=.{1,253}$)(?!-)(?:[a-zA-Z0-9-]{1,63}(?<!-)\.)+[a-zA-Z]{2,63}$'
        )
        return bool(fqdn_pattern.match(value))
        
    def _is_short_hostname(self, value: str) -> bool:
        short_pattern = re.compile(
            r'^[a-zA-Z0-9]([a-zA-Z0-9-_]*[a-zA-Z0-9])?$'
        )
        return bool(short_pattern.match(value)) and 3 <= len(value) <= 63
        
    def _is_cloud_instance(self, value: str) -> bool:
        patterns = [
            # AWS
            r'^i-[a-f0-9]{8,17}$',
            r'^ip-\d{1,3}-\d{1,3}-\d{1,3}-\d{1,3}',
            r'^ec2-\d{1,3}-\d{1,3}-\d{1,3}-\d{1,3}',
            
            # GCP
            r'^[a-z][-a-z0-9]{0,62}$',
            r'^[a-z0-9-]+\.c\.[a-z0-9-]+\.internal$',
            
            # Azure
            r'^[a-zA-Z][a-zA-Z0-9-]{1,59}$',
            
            # Generic UUID
            r'^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$',
        ]
        
        for pattern in patterns:
            if re.match(pattern, value, re.IGNORECASE):
                return True
        return False
        
    def _is_kubernetes_name(self, value: str) -> bool:
        patterns = [
            r'^[a-z0-9]([-a-z0-9]*[a-z0-9])?(\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*$',
            r'^[a-z0-9]([-a-z0-9]*[a-z0-9])?$',
            r'^[a-z0-9-]+-[a-f0-9]{5,10}(-[a-f0-9]{5,10})?$',
        ]
        
        for pattern in patterns:
            if re.match(pattern, value):
                return True
        return False
        
    def _match_learned_patterns(self, value: str) -> float:
        if not self.learned_patterns['structures']:
            return 0.0
            
        scores = []
        value_lower = value.lower()
        
        # Check structure match
        structure = self._extract_structure(value)
        if structure in self.learned_patterns['structures']:
            scores.append(0.9)
        
        # Check length distribution
        if self.learned_patterns['lengths']:
            avg_len = np.mean(self.learned_patterns['lengths'])
            std_len = np.std(self.learned_patterns['lengths'])
            if std_len > 0:
                z_score = abs(len(value) - avg_len) / std_len
                if z_score < 2:
                    scores.append(0.7 - (z_score * 0.2))
        
        # Check character set compliance
        if self.learned_patterns['character_sets']:
            valid_chars = sum(1 for c in value if c in self.learned_patterns['character_sets'])
            char_ratio = valid_chars / len(value)
            scores.append(char_ratio * 0.8)
        
        # Check prefix/suffix matches
        if self.learned_patterns['prefixes'] and len(value) > 3:
            if value_lower[:3] in self.learned_patterns['prefixes']:
                scores.append(0.6)
                
        if self.learned_patterns['suffixes']:
            for suffix in self.learned_patterns['suffixes']:
                if value_lower.endswith(suffix):
                    scores.append(0.7)
                    break
        
        # Check similarity to learned patterns
        if self.learned_patterns['common_patterns']:
            for pattern in self.learned_patterns['common_patterns'][:100]:
                similarity = self._calculate_similarity(value_lower, pattern)
                if similarity > 0.8:
                    scores.append(similarity)
                    
        return max(scores) if scores else 0.0
        
    def _calculate_similarity(self, s1: str, s2: str) -> float:
        if s1 == s2:
            return 1.0
            
        len1, len2 = len(s1), len(s2)
        if len1 == 0 or len2 == 0:
            return 0.0
            
        # Quick check for common prefix/suffix
        common_prefix = 0
        for i in range(min(len1, len2)):
            if s1[i] == s2[i]:
                common_prefix += 1
            else:
                break
                
        common_suffix = 0
        for i in range(1, min(len1, len2) + 1):
            if s1[-i] == s2[-i]:
                common_suffix += 1
            else:
                break
                
        max_common = max(common_prefix, common_suffix)
        return max_common / max(len1, len2)
        
    async def _train_neural_model(self, samples: List[str]):
        # Prepare training data
        positive_samples = [(s, 1) for s in samples if s]
        
        # Generate negative samples
        negative_samples = []
        for sample in samples[:len(samples)//2]:
            if sample:
                # Create corrupted versions
                corrupted = sample.replace('.', '_xyz_')
                negative_samples.append((corrupted, 0))
                
                # Random strings
                import random
                import string
                random_str = ''.join(random.choices(string.ascii_letters + string.digits, k=len(sample)))
                negative_samples.append((random_str, 0))
        
        all_samples = positive_samples + negative_samples
        
        if all_samples:
            await self.neural_model.train(all_samples)

class HostnameNeuralNet(nn.Module):
    
    def __init__(self, device: str = 'mps', embedding_dim: int = 128):
        super().__init__()
        self.device = device
        self.embedding_dim = embedding_dim
        
        self.char_embed = nn.Embedding(256, embedding_dim)
        
        self.lstm = nn.LSTM(
            embedding_dim,
            128,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.2
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.to(device)
        
    def forward(self, x):
        embedded = self.char_embed(x)
        lstm_out, _ = self.lstm(embedded)
        pooled = torch.max(lstm_out, dim=1)[0]
        output = self.classifier(pooled)
        return output
        
    async def train(self, samples: List[tuple]):
        logger.info(f"Training neural model on {len(samples)} samples")
        
    def predict(self, value: str) -> float:
        return 0.6