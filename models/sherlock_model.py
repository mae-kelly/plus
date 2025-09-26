import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any
import re
from collections import Counter

class SherlockModel(nn.Module):
    def __init__(self, input_dim: int = 1588, num_types: int = 78, device: str = 'mps'):
        super().__init__()
        self.device = device
        
        # Enhanced feature encoder with residual connections
        self.feature_encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Multi-head attention for better feature understanding
        self.attention = nn.MultiheadAttention(256, 4, dropout=0.1, batch_first=True)
        
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_types)
        )
        
        self.confidence_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.to(device)
    
    def forward(self, x):
        features = self.feature_encoder(x)
        
        # Add self-attention
        features_expanded = features.unsqueeze(1)
        attended_features, _ = self.attention(features_expanded, features_expanded, features_expanded)
        features = attended_features.squeeze(1)
        
        logits = self.classifier(features)
        confidence = self.confidence_head(features)
        return logits, confidence
    
    def extract_features(self, column_name: str, values: List[Any]) -> np.ndarray:
        """Extract comprehensive statistical and pattern features"""
        features = []
        
        str_values = [str(v) if v is not None else '' for v in values]
        
        # Statistical features
        lengths = [len(v) for v in str_values]
        features.extend(self._get_statistical_features(lengths))
        
        # Character-level features
        char_features = self._extract_char_features(str_values)
        features.extend(char_features)
        
        # Pattern features
        pattern_features = self._extract_pattern_features(str_values)
        features.extend(pattern_features)
        
        # Semantic features from column name
        semantic_features = self._extract_semantic_features(column_name)
        features.extend(semantic_features)
        
        # Entropy and distribution features
        entropy_features = self._extract_entropy_features(str_values)
        features.extend(entropy_features)
        
        # N-gram features
        ngram_features = self._extract_ngram_features(str_values)
        features.extend(ngram_features)
        
        # Pad or truncate to input_dim
        features = np.array(features[:1588] + [0] * max(0, 1588 - len(features)), dtype=np.float32)
        
        return features
    
    def _get_statistical_features(self, lengths: List[int]) -> List[float]:
        if not lengths:
            return [0] * 15
        
        return [
            np.mean(lengths),
            np.median(lengths),
            np.std(lengths) if len(lengths) > 1 else 0,
            np.min(lengths),
            np.max(lengths),
            np.percentile(lengths, 25) if lengths else 0,
            np.percentile(lengths, 75) if lengths else 0,
            np.percentile(lengths, 10) if lengths else 0,
            np.percentile(lengths, 90) if lengths else 0,
            len(set(lengths)),
            max(Counter(lengths).values()) if lengths else 0,
            len([l for l in lengths if l == 0]),
            len([l for l in lengths if l > 0]),
            np.var(lengths) if len(lengths) > 1 else 0,
            max(lengths) - min(lengths) if lengths else 0
        ]
    
    def _extract_char_features(self, values: List[str]) -> List[float]:
        features = []
        
        all_chars = ''.join(values)
        total_chars = max(len(all_chars), 1)
        
        # Basic character type ratios
        features.append(sum(c.isalpha() for c in all_chars) / total_chars)
        features.append(sum(c.isdigit() for c in all_chars) / total_chars)
        features.append(sum(c.isspace() for c in all_chars) / total_chars)
        features.append(sum(c.isupper() for c in all_chars) / total_chars)
        features.append(sum(c.islower() for c in all_chars) / total_chars)
        features.append(sum(c.isalnum() for c in all_chars) / total_chars)
        
        # Special character frequencies
        special_chars = '!@#$%^&*()[]{}|\\:;"\'<>,.?/~`-_+=\t\n'
        for char in special_chars:
            features.append(all_chars.count(char) / total_chars)
        
        # Unicode categories
        features.append(sum(ord(c) > 127 for c in all_chars) / total_chars)
        features.append(sum(c.isdecimal() for c in all_chars) / total_chars)
        
        return features[:80]
    
    def _extract_pattern_features(self, values: List[str]) -> List[float]:
        features = []
        
        patterns = {
            'email': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
            'url': r'^https?://[^\s]+$',
            'ipv4': r'^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$',
            'ipv6': r'^(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$',
            'uuid': r'^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$',
            'date_iso': r'^\d{4}-\d{2}-\d{2}',
            'datetime_iso': r'^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}',
            'phone_us': r'^(\+1)?[\s.-]?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}$',
            'hostname': r'^[a-zA-Z0-9][a-zA-Z0-9-]{0,61}[a-zA-Z0-9]?$',
            'fqdn': r'^[a-zA-Z0-9][a-zA-Z0-9-]{0,61}[a-zA-Z0-9]?(\.[a-zA-Z0-9][a-zA-Z0-9-]{0,61}[a-zA-Z0-9]?)*$',
            'mac_address': r'^([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})$',
            'md5': r'^[a-f0-9]{32}$',
            'sha1': r'^[a-f0-9]{40}$',
            'sha256': r'^[a-f0-9]{64}$',
            'base64': r'^[A-Za-z0-9+/]+=*$',
            'json': r'^\{.*\}$|^\[.*\]$',
            'xml': r'^<.*>.*</.*>$',
            'credit_card': r'^\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}$',
            'ssn': r'^\d{3}-\d{2}-\d{4}$',
            'zip_us': r'^\d{5}(-\d{4})?$',
            'hex_color': r'^#[0-9A-Fa-f]{6}$',
            'aws_arn': r'^arn:aws:[a-z0-9-]+:[a-z0-9-]*:\d{12}:',
            'docker_image': r'^[a-z0-9]+([._-][a-z0-9]+)*(/[a-z0-9]+([._-][a-z0-9]+)*)*',
            'semver': r'^\d+\.\d+\.\d+',
            'jwt': r'^[A-Za-z0-9-_]+\.[A-Za-z0-9-_]+\.[A-Za-z0-9-_]+$'
        }
        
        for pattern_name, pattern_regex in patterns.items():
            try:
                matches = sum(1 for v in values if re.match(pattern_regex, v))
                features.append(matches / max(len(values), 1))
            except:
                features.append(0)
        
        return features[:200]
    
    def _extract_semantic_features(self, column_name: str) -> List[float]:
        features = []
        
        name_lower = column_name.lower()
        
        # Semantic keyword indicators
        semantic_indicators = [
            'id', 'identifier', 'key', 'code', 'num', 'number',
            'name', 'title', 'label', 'description', 'desc',
            'email', 'mail', 'address', 'addr',
            'phone', 'tel', 'mobile', 'cell',
            'date', 'time', 'timestamp', 'created', 'modified', 'updated',
            'amount', 'price', 'cost', 'value', 'balance', 'total', 'sum',
            'count', 'quantity', 'qty',
            'host', 'hostname', 'server', 'machine', 'node', 'instance',
            'ip', 'ipaddr', 'ip_address',
            'domain', 'url', 'uri', 'path', 'endpoint',
            'user', 'username', 'customer', 'client', 'person', 'owner',
            'status', 'state', 'flag', 'active', 'enabled',
            'type', 'kind', 'category', 'class',
            'env', 'environment', 'region', 'zone', 'datacenter', 'dc',
            'app', 'application', 'service', 'component',
            'version', 'ver', 'revision', 'build',
            'error', 'exception', 'warning', 'info', 'debug',
            'config', 'configuration', 'setting', 'param', 'parameter'
        ]
        
        for indicator in semantic_indicators:
            features.append(float(indicator in name_lower))
        
        # Column name structure features
        features.append(len(column_name))
        features.append(column_name.count('_'))
        features.append(column_name.count('-'))
        features.append(column_name.count('.'))
        features.append(float(column_name.isupper()))
        features.append(float(column_name.islower()))
        features.append(float(column_name[0].isupper() if column_name else 0))
        features.append(float(any(c.isdigit() for c in column_name)))
        
        # Naming convention detection
        features.append(float('_' in column_name))  # snake_case
        features.append(float(column_name[0].islower() and any(c.isupper() for c in column_name[1:])))  # camelCase
        features.append(float(column_name[0].isupper() and any(c.isupper() for c in column_name[1:])))  # PascalCase
        
        return features[:150]
    
    def _extract_entropy_features(self, values: List[str]) -> List[float]:
        features = []
        
        # Character entropy
        all_chars = ''.join(values)
        if all_chars:
            char_counts = Counter(all_chars)
            total = len(all_chars)
            char_entropy = -sum((count/total) * np.log2(count/total) for count in char_counts.values())
            features.append(char_entropy)
        else:
            features.append(0)
        
        # Value entropy
        if values:
            value_counts = Counter(values)
            total = len(values)
            value_entropy = -sum((count/total) * np.log2(count/total) for count in value_counts.values() if count > 0)
            features.append(value_entropy)
        else:
            features.append(0)
        
        # Uniqueness ratio
        features.append(len(set(values)) / max(len(values), 1))
        
        # Gini coefficient
        if values:
            counts = list(Counter(values).values())
            sorted_counts = sorted(counts)
            n = len(sorted_counts)
            if n > 0 and sum(sorted_counts) > 0:
                cumsum = np.cumsum(sorted_counts)
                gini = (2 * np.sum((np.arange(n) + 1) * sorted_counts)) / (n * cumsum[-1]) - (n + 1) / n
                features.append(gini)
            else:
                features.append(0)
        else:
            features.append(0)
        
        return features[:20]
    
    def _extract_ngram_features(self, values: List[str]) -> List[float]:
        """Extract n-gram based features"""
        features = []
        
        # Common bigrams and trigrams
        all_text = ' '.join(values[:100])  # Sample for performance
        
        if len(all_text) >= 2:
            bigrams = [all_text[i:i+2] for i in range(len(all_text)-1)]
            bigram_counts = Counter(bigrams)
            
            # Top bigram frequencies
            top_bigrams = bigram_counts.most_common(10)
            for bigram, count in top_bigrams:
                features.append(count / max(len(bigrams), 1))
            
            # Pad if needed
            while len(features) < 10:
                features.append(0)
        else:
            features.extend([0] * 10)
        
        if len(all_text) >= 3:
            trigrams = [all_text[i:i+3] for i in range(len(all_text)-2)]
            trigram_counts = Counter(trigrams)
            
            # Top trigram frequencies
            top_trigrams = trigram_counts.most_common(10)
            for trigram, count in top_trigrams:
                features.append(count / max(len(trigrams), 1))
            
            # Pad if needed
            while len(features) < 20:
                features.append(0)
        else:
            features.extend([0] * 10)
        
        return features[:20]