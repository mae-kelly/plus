import torch
import torch.nn as nn
import numpy as np
from typing import List, Any, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
import hashlib
from collections import Counter
import re

class SimpleEmbedder(nn.Module):
    """Simple embedding model to replace SentenceTransformer"""
    def __init__(self, vocab_size: int = 5000, embed_dim: int = 384, device: str = 'mps'):
        super().__init__()
        self.device = device
        
        # Character-level CNN
        self.char_embed = nn.Embedding(256, 32)
        self.conv1 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.pool = nn.AdaptiveMaxPool1d(1)
        
        # Final projection
        self.projection = nn.Linear(128, embed_dim)
        
        self.to(device)
    
    def forward(self, text: str) -> torch.Tensor:
        # Convert text to character indices
        chars = [ord(c) % 256 for c in text[:500]]
        if not chars:
            return torch.zeros(1, 384, device=self.device)
        
        char_tensor = torch.tensor(chars, device=self.device).unsqueeze(0)
        
        # Embed and convolve
        x = self.char_embed(char_tensor).transpose(1, 2)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)
        
        # Project to final dimension
        embedding = self.projection(x)
        
        return embedding

class AdvancedFeatureExtractor:
    def __init__(self, device: str = 'mps'):
        self.device = device
        
        # Use simple embedder instead of SentenceTransformer
        self.embedder = SimpleEmbedder(device=device)
        self.embedder.eval()
        
        self.tfidf = TfidfVectorizer(max_features=500, ngram_range=(1, 3))
        self.feature_cache = {}
        self.pattern_cache = {}
        
    async def extract(self, column_name: str, values: List[Any]) -> np.ndarray:
        cache_key = self._get_cache_key(column_name, values[:10])
        
        if cache_key in self.feature_cache:
            return self.feature_cache[cache_key]
        
        features = []
        
        # Statistical features
        statistical_features = self._extract_statistical_features(values)
        features.extend(statistical_features)
        
        # Pattern features
        pattern_features = self._extract_pattern_features(values)
        features.extend(pattern_features)
        
        # Semantic features
        semantic_features = self._extract_semantic_features(column_name, values)
        features.extend(semantic_features)
        
        # Distribution features
        distribution_features = self._extract_distribution_features(values)
        features.extend(distribution_features)
        
        # Embedding features (using simple embedder)
        embedding_features = self._extract_embedding_features(column_name, values)
        features.extend(embedding_features)
        
        # Pad or truncate to 1588
        features = np.array(features, dtype=np.float32)
        
        if len(features) < 1588:
            features = np.pad(features, (0, 1588 - len(features)), mode='constant')
        elif len(features) > 1588:
            features = features[:1588]
        
        self.feature_cache[cache_key] = features
        return features
    
    def _extract_statistical_features(self, values: List[Any]) -> List[float]:
        features = []
        
        str_values = [str(v) if v is not None else '' for v in values]
        lengths = [len(v) for v in str_values]
        
        if lengths:
            features.extend([
                np.mean(lengths),
                np.median(lengths),
                np.std(lengths),
                np.min(lengths),
                np.max(lengths),
                np.percentile(lengths, 25),
                np.percentile(lengths, 75),
                len(set(lengths))
            ])
        else:
            features.extend([0] * 8)
        
        # Unique ratio
        unique_ratio = len(set(str_values)) / max(len(str_values), 1)
        features.append(unique_ratio)
        
        # Null ratio
        null_ratio = sum(1 for v in values if v is None or v == '') / max(len(values), 1)
        features.append(null_ratio)
        
        # Numeric values
        numeric_values = []
        for v in values:
            try:
                numeric_values.append(float(v))
            except:
                pass
        
        if numeric_values:
            features.extend([
                np.mean(numeric_values),
                np.std(numeric_values),
                np.min(numeric_values),
                np.max(numeric_values),
                len(numeric_values) / len(values)
            ])
        else:
            features.extend([0] * 5)
        
        return features[:50]
    
    def _extract_pattern_features(self, values: List[Any]) -> List[float]:
        features = []
        str_values = [str(v) if v is not None else '' for v in values]
        
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
            'credit_card': r'^\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}$'
        }
        
        for pattern_name, pattern_regex in patterns.items():
            matches = sum(1 for v in str_values if re.match(pattern_regex, v))
            features.append(matches / max(len(str_values), 1))
        
        # Character class features
        char_features = self._extract_char_class_features(str_values)
        features.extend(char_features)
        
        return features[:100]
    
    def _extract_char_class_features(self, values: List[str]) -> List[float]:
        features = []
        
        all_text = ''.join(values)
        if not all_text:
            return [0] * 20
        
        total_chars = len(all_text)
        
        features.append(sum(c.isalpha() for c in all_text) / total_chars)
        features.append(sum(c.isdigit() for c in all_text) / total_chars)
        features.append(sum(c.isspace() for c in all_text) / total_chars)
        features.append(sum(c.isupper() for c in all_text) / total_chars)
        features.append(sum(c.islower() for c in all_text) / total_chars)
        features.append(sum(c in '.-_' for c in all_text) / total_chars)
        features.append(sum(c in '@#$%' for c in all_text) / total_chars)
        features.append(sum(c in '()[]{}' for c in all_text) / total_chars)
        features.append(sum(c in '/\\|' for c in all_text) / total_chars)
        features.append(sum(c in ',:;' for c in all_text) / total_chars)
        
        # Non-ASCII
        features.append(sum(ord(c) > 127 for c in all_text) / total_chars)
        
        # Entropy
        char_entropy = self._calculate_entropy(all_text)
        features.append(char_entropy)
        
        # Bigram entropy
        bigram_entropy = self._calculate_bigram_entropy(all_text)
        features.append(bigram_entropy)
        
        return features[:20]
    
    def _extract_semantic_features(self, column_name: str, values: List[Any]) -> List[float]:
        features = []
        
        name_lower = column_name.lower()
        
        # Semantic indicators
        semantic_indicators = [
            'id', 'name', 'email', 'phone', 'address', 'date', 'time',
            'amount', 'price', 'cost', 'value', 'count', 'number',
            'host', 'server', 'ip', 'domain', 'url', 'path',
            'user', 'customer', 'client', 'owner', 'created', 'modified'
        ]
        
        for indicator in semantic_indicators:
            features.append(float(indicator in name_lower))
        
        # Column name structure
        features.append(len(column_name))
        features.append(column_name.count('_'))
        features.append(float(column_name.isupper()))
        features.append(float(column_name.islower()))
        
        return features[:100]
    
    def _extract_distribution_features(self, values: List[Any]) -> List[float]:
        features = []
        
        str_values = [str(v) if v is not None else '' for v in values]
        
        if not str_values:
            return [0] * 50
        
        value_counts = Counter(str_values)
        frequencies = list(value_counts.values())
        
        if frequencies:
            features.append(max(frequencies))
            features.append(min(frequencies))
            features.append(np.mean(frequencies))
            features.append(np.std(frequencies))
            
            # Top 10 frequency
            top_10_freq = sum(sorted(frequencies, reverse=True)[:10])
            features.append(top_10_freq / sum(frequencies))
            
            # Gini coefficient
            gini = self._calculate_gini_coefficient(frequencies)
            features.append(gini)
        else:
            features.extend([0] * 6)
        
        # Length distribution
        lengths = [len(v) for v in str_values]
        if lengths:
            length_counts = Counter(lengths)
            length_entropy = self._calculate_entropy_from_counts(length_counts)
            features.append(length_entropy)
            
            # Normality test
            from scipy import stats
            if len(set(lengths)) > 1:
                _, p_value = stats.normaltest(lengths)
                features.append(p_value)
            else:
                features.append(1.0)
        else:
            features.extend([0, 1.0])
        
        return features[:50]
    
    def _extract_embedding_features(self, column_name: str, values: List[Any]) -> np.ndarray:
        """Extract embedding using simple embedder"""
        sample_text = f"{column_name} " + " ".join([str(v) for v in values[:20] if v])
        
        with torch.no_grad():
            embedding = self.embedder(sample_text)
        
        return embedding.cpu().numpy().flatten()[:384]
    
    def _calculate_entropy(self, text: str) -> float:
        if not text:
            return 0
        
        char_counts = Counter(text)
        total = len(text)
        entropy = -sum((count/total) * np.log2(count/total) for count in char_counts.values())
        
        return entropy
    
    def _calculate_bigram_entropy(self, text: str) -> float:
        if len(text) < 2:
            return 0
        
        bigrams = [text[i:i+2] for i in range(len(text)-1)]
        bigram_counts = Counter(bigrams)
        total = len(bigrams)
        
        entropy = -sum((count/total) * np.log2(count/total) for count in bigram_counts.values())
        
        return entropy
    
    def _calculate_entropy_from_counts(self, counts: Counter) -> float:
        total = sum(counts.values())
        if total == 0:
            return 0
        
        entropy = -sum((count/total) * np.log2(count/total) for count in counts.values())
        
        return entropy
    
    def _calculate_gini_coefficient(self, values: List[float]) -> float:
        sorted_values = sorted(values)
        n = len(sorted_values)
        
        if n == 0 or sum(sorted_values) == 0:
            return 0
        
        cumsum = np.cumsum(sorted_values)
        gini = (2 * np.sum((np.arange(n) + 1) * sorted_values)) / (n * cumsum[-1]) - (n + 1) / n
        
        return gini
    
    def _get_cache_key(self, column_name: str, values: List[Any]) -> str:
        value_str = str(values[:5])
        key_str = f"{column_name}:{value_str}"
        return hashlib.md5(key_str.encode()).hexdigest()