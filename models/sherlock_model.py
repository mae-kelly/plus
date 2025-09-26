import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any

class SherlockModel(nn.Module):
    def __init__(self, input_dim: int = 1588, num_types: int = 78, device: str = 'mps'):
        super().__init__()
        self.device = device
        
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
        logits = self.classifier(features)
        confidence = self.confidence_head(features)
        return logits, confidence
    
    def extract_features(self, column_name: str, values: List[Any]) -> np.ndarray:
        features = []
        
        str_values = [str(v) if v is not None else '' for v in values]
        
        lengths = [len(v) for v in str_values]
        features.extend([
            np.mean(lengths) if lengths else 0,
            np.median(lengths) if lengths else 0,
            np.std(lengths) if lengths else 0,
            np.min(lengths) if lengths else 0,
            np.max(lengths) if lengths else 0
        ])
        
        char_features = self._extract_char_features(str_values)
        features.extend(char_features)
        
        pattern_features = self._extract_pattern_features(str_values)
        features.extend(pattern_features)
        
        semantic_features = self._extract_semantic_features(column_name)
        features.extend(semantic_features)
        
        features = np.array(features[:1588] + [0] * (1588 - len(features)), dtype=np.float32)
        
        return features
    
    def _extract_char_features(self, values: List[str]) -> List[float]:
        features = []
        
        all_chars = ''.join(values)
        total_chars = len(all_chars)
        
        if total_chars == 0:
            return [0] * 50
        
        features.append(sum(c.isalpha() for c in all_chars) / total_chars)
        features.append(sum(c.isdigit() for c in all_chars) / total_chars)
        features.append(sum(c.isspace() for c in all_chars) / total_chars)
        features.append(sum(c.isupper() for c in all_chars) / total_chars)
        features.append(sum(c.islower() for c in all_chars) / total_chars)
        
        special_chars = '!@#$%^&*()[]{}|\\:;"<>,.?/~`'
        for char in special_chars:
            features.append(all_chars.count(char) / total_chars)
        
        return features[:50]
    
    def _extract_pattern_features(self, values: List[str]) -> List[float]:
        import re
        features = []
        
        patterns = {
            'email': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
            'url': r'^https?://[^\s]+$',
            'ip': r'^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$',
            'uuid': r'^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$',
            'date': r'\d{4}-\d{2}-\d{2}',
            'phone': r'^\+?1?\d{9,15}$',
            'hostname': r'^[a-zA-Z0-9][a-zA-Z0-9-]{0,61}[a-zA-Z0-9]?$'
        }
        
        for pattern_name, pattern in patterns.items():
            matches = sum(1 for v in values if re.match(pattern, v))
            features.append(matches / max(len(values), 1))
        
        return features[:100]
    
    def _extract_semantic_features(self, column_name: str) -> List[float]:
        features = []
        
        name_lower = column_name.lower()
        
        semantic_indicators = [
            'id', 'name', 'email', 'phone', 'address', 'date', 'time',
            'amount', 'price', 'cost', 'value', 'count', 'number',
            'host', 'server', 'ip', 'domain', 'url', 'path',
            'user', 'customer', 'client', 'owner', 'created', 'modified'
        ]
        
        for indicator in semantic_indicators:
            features.append(float(indicator in name_lower))
        
        features.append(len(column_name))
        features.append(column_name.count('_'))
        features.append(float(column_name.isupper()))
        features.append(float(column_name.islower()))
        
        return features[:100]