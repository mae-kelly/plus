import torch
import torch.nn as nn
import numpy as np
from typing import List, Any, Dict
import hashlib
from collections import Counter
import re

class LocalEmbeddingModel(nn.Module):
    """Local embedding model without external dependencies"""
    def __init__(self, vocab_size: int = 10000, embedding_dim: int = 384, device: str = 'mps'):
        super().__init__()
        self.device = device
        self.embedding_dim = embedding_dim
        
        # Character-level embeddings
        self.char_embedding = nn.Embedding(256, 64)  # ASCII characters
        
        # Word embeddings
        self.word_embedding = nn.Embedding(vocab_size, 128)
        
        # Convolutional layers for different n-grams
        self.conv1 = nn.Conv1d(64, 128, kernel_size=1, padding=0)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=2, padding=0)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=0)
        
        # Final projection
        self.projection = nn.Sequential(
            nn.Linear(384 + 128, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, embedding_dim)
        )
        
        self.to(device)
    
    def forward(self, text: str) -> torch.Tensor:
        """Generate embedding for text"""
        # Character-level processing
        char_ids = torch.tensor([ord(c) if ord(c) < 256 else 0 for c in text[:512]], 
                                device=self.device).unsqueeze(0)
        
        if char_ids.shape[1] == 0:
            return torch.zeros(1, self.embedding_dim, device=self.device)
        
        char_embeds = self.char_embedding(char_ids).transpose(1, 2)
        
        # Apply convolutions
        conv_outputs = []
        
        if char_embeds.shape[2] >= 1:
            conv1_out = torch.max(self.conv1(char_embeds), dim=2)[0]
            conv_outputs.append(conv1_out)
        
        if char_embeds.shape[2] >= 2:
            conv2_out = torch.max(self.conv2(char_embeds), dim=2)[0]
            conv_outputs.append(conv2_out)
        
        if char_embeds.shape[2] >= 3:
            conv3_out = torch.max(self.conv3(char_embeds), dim=2)[0]
            conv_outputs.append(conv3_out)
        
        if not conv_outputs:
            conv_outputs = [torch.zeros(1, 128, device=self.device)]
        
        # Concatenate convolution outputs
        conv_features = torch.cat(conv_outputs, dim=1)
        
        # Add padding if needed
        if conv_features.shape[1] < 384:
            padding = torch.zeros(1, 384 - conv_features.shape[1], device=self.device)
            conv_features = torch.cat([conv_features, padding], dim=1)
        elif conv_features.shape[1] > 384:
            conv_features = conv_features[:, :384]
        
        # Word-level features (simple hash-based)
        words = text.lower().split()[:20]
        word_ids = [hash(w) % 10000 for w in words]
        
        if word_ids:
            word_tensor = torch.tensor(word_ids, device=self.device).unsqueeze(0)
            word_embeds = self.word_embedding(word_tensor).mean(dim=1)
        else:
            word_embeds = torch.zeros(1, 128, device=self.device)
        
        # Combine features
        combined = torch.cat([conv_features, word_embeds], dim=1)
        
        # Final projection
        embedding = self.projection(combined)
        
        return embedding

class AdvancedFeatureExtractor:
    def __init__(self, device: str = 'mps'):
        self.device = device
        
        # Initialize local embedding model
        self.embedding_model = LocalEmbeddingModel(device=device)
        self.embedding_model.eval()
        
        # TF-IDF implementation
        self.vocabulary = {}
        self.idf_weights = {}
        
        self.feature_cache = {}
        self.pattern_cache = {}
        
    async def extract(self, column_name: str, values: List[Any]) -> np.ndarray:
        """Extract comprehensive features from column values"""
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
        
        # Embedding features
        embedding_features = self._extract_embedding_features(column_name, values)
        features.extend(embedding_features)
        
        # Character-level features
        char_features = self._extract_char_level_features(values)
        features.extend(char_features)
        
        # Ensure fixed size
        features = np.array(features, dtype=np.float32)
        
        if len(features) < 1588:
            features = np.pad(features, (0, 1588 - len(features)), mode='constant')
        elif len(features) > 1588:
            features = features[:1588]
        
        self.feature_cache[cache_key] = features
        return features
    
    def _extract_statistical_features(self, values: List[Any]) -> List[float]:
        """Extract statistical features"""
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
                len(set(lengths)),
                np.var(lengths) if len(lengths) > 1 else 0,
                max(lengths) - min(lengths) if lengths else 0
            ])
        else:
            features.extend([0] * 10)
        
        # Uniqueness metrics
        unique_ratio = len(set(str_values)) / max(len(str_values), 1)
        features.append(unique_ratio)
        
        # Null ratio
        null_ratio = sum(1 for v in values if v is None or v == '') / max(len(values), 1)
        features.append(null_ratio)
        
        # Numeric statistics
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
                len(numeric_values) / len(values),
                np.median(numeric_values),
                np.percentile(numeric_values, 10) if numeric_values else 0,
                np.percentile(numeric_values, 90) if numeric_values else 0
            ])
        else:
            features.extend([0] * 8)
        
        return features[:50]
    
    def _extract_pattern_features(self, values: List[Any]) -> List[float]:
        """Extract pattern-based features"""
        features = []
        str_values = [str(v) if v is not None else '' for v in values]
        
        patterns = {
            'email': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,},
            'url': r'^https?://[^\s]+,
            'ipv4': r'^(?:[0-9]{1,3}\.){3}[0-9]{1,3},
            'ipv6': r'^(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4},
            'uuid': r'^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12},
            'date_iso': r'^\d{4}-\d{2}-\d{2}',
            'datetime_iso': r'^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}',
            'phone_us': r'^(\+1)?[\s.-]?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4},
            'hostname': r'^[a-zA-Z0-9][a-zA-Z0-9-]{0,61}[a-zA-Z0-9]?,
            'fqdn': r'^[a-zA-Z0-9][a-zA-Z0-9-]{0,61}[a-zA-Z0-9]?(\.[a-zA-Z0-9][a-zA-Z0-9-]{0,61}[a-zA-Z0-9]?)*,
            'mac_address': r'^([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2}),
            'md5': r'^[a-f0-9]{32},
            'sha1': r'^[a-f0-9]{40},
            'sha256': r'^[a-f0-9]{64},
            'base64': r'^[A-Za-z0-9+/]+=*,
            'json': r'^\{.*\}$|^\[.*\],
            'xml': r'^<.*>.*</.*>,
            'credit_card': r'^\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}
        }
        
        for pattern_name, pattern_regex in patterns.items():
            try:
                matches = sum(1 for v in str_values if re.match(pattern_regex, v))
                features.append(matches / max(len(str_values), 1))
            except:
                features.append(0)
        
        # Add character class features
        char_features = self._extract_char_class_features(str_values)
        features.extend(char_features)
        
        return features[:100]
    
    def _extract_char_class_features(self, values: List[str]) -> List[float]:
        """Extract character class features"""
        features = []
        
        all_text = ''.join(values)
        if not all_text:
            return [0] * 20
        
        total_chars = len(all_text)
        
        # Basic character types
        features.append(sum(c.isalpha() for c in all_text) / total_chars)
        features.append(sum(c.isdigit() for c in all_text) / total_chars)
        features.append(sum(c.isspace() for c in all_text) / total_chars)
        features.append(sum(c.isupper() for c in all_text) / total_chars)
        features.append(sum(c.islower() for c in all_text) / total_chars)
        
        # Special characters
        features.append(sum(c in '.-_' for c in all_text) / total_chars)
        features.append(sum(c in '@#$%' for c in all_text) / total_chars)
        features.append(sum(c in '()[]{}' for c in all_text) / total_chars)
        features.append(sum(c in '/\\|' for c in all_text) / total_chars)
        features.append(sum(c in ',:;' for c in all_text) / total_chars)
        
        # Unicode
        features.append(sum(ord(c) > 127 for c in all_text) / total_chars)
        
        # Entropy
        char_entropy = self._calculate_entropy(all_text)
        features.append(char_entropy)
        
        # Bigram entropy
        bigram_entropy = self._calculate_bigram_entropy(all_text)
        features.append(bigram_entropy)
        
        return features[:20]
    
    def _extract_semantic_features(self, column_name: str, values: List[Any]) -> List[float]:
        """Extract semantic features from column name"""
        features = []
        
        name_lower = column_name.lower()
        
        # Semantic categories
        semantic_categories = {
            'identifier': ['id', 'key', 'code', 'number', 'num', 'no'],
            'name': ['name', 'title', 'label', 'description'],
            'location': ['address', 'location', 'city', 'state', 'country', 'zip'],
            'temporal': ['date', 'time', 'timestamp', 'created', 'modified'],
            'network': ['host', 'server', 'ip', 'domain', 'url', 'port'],
            'user': ['user', 'customer', 'client', 'person', 'owner'],
            'financial': ['amount', 'price', 'cost', 'value', 'balance'],
            'status': ['status', 'state', 'flag', 'active', 'enabled']
        }
        
        for category, keywords in semantic_categories.items():
            score = sum(1 for kw in keywords if kw in name_lower) / len(keywords)
            features.append(score)
        
        # Name structure
        name_tokens = re.split(r'[_\-\.\s]+', name_lower)
        features.append(len(name_tokens))
        features.append(np.mean([len(t) for t in name_tokens]) if name_tokens else 0)
        
        # Naming conventions
        features.append(float('_' in column_name))
        features.append(float('-' in column_name))
        features.append(float('.' in column_name))
        features.append(float(column_name.isupper()))
        features.append(float(column_name.islower()))
        
        return features[:50]
    
    def _extract_distribution_features(self, values: List[Any]) -> List[float]:
        """Extract distribution features"""
        features = []
        
        str_values = [str(v) if v is not None else '' for v in values]
        
        if not str_values:
            return [0] * 50
        
        # Value frequency distribution
        value_counts = Counter(str_values)
        frequencies = list(value_counts.values())
        
        if frequencies:
            features.append(max(frequencies))
            features.append(min(frequencies))
            features.append(np.mean(frequencies))
            features.append(np.std(frequencies))
            
            # Top frequency ratio
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
        else:
            features.append(0)
        
        # Fill to fixed size
        while len(features) < 50:
            features.append(0)
        
        return features[:50]
    
    def _extract_embedding_features(self, column_name: str, values: List[Any]) -> np.ndarray:
        """Extract embedding features using local model"""
        # Create sample text
        sample_text = f"{column_name} " + " ".join([str(v) for v in values[:20] if v])
        
        with torch.no_grad():
            embedding = self.embedding_model(sample_text)
        
        return embedding.cpu().numpy().flatten()[:384]
    
    def _extract_char_level_features(self, values: List[str]) -> List[float]:
        """Extract character-level n-gram features"""
        features = []
        
        # Sample text
        sample_text = ' '.join([str(v) for v in values[:50] if v])[:1000]
        
        if not sample_text:
            return [0] * 50
        
        # Character bigrams
        bigrams = [sample_text[i:i+2] for i in range(len(sample_text)-1)]
        bigram_counts = Counter(bigrams)
        
        # Top bigram frequencies
        top_bigrams = bigram_counts.most_common(20)
        for bigram, count in top_bigrams:
            features.append(count / len(bigrams) if bigrams else 0)
        
        # Character trigrams
        trigrams = [sample_text[i:i+3] for i in range(len(sample_text)-2)]
        trigram_counts = Counter(trigrams)
        
        # Top trigram frequencies
        top_trigrams = trigram_counts.most_common(20)
        for trigram, count in top_trigrams:
            features.append(count / len(trigrams) if trigrams else 0)
        
        # Pad to fixed size
        while len(features) < 50:
            features.append(0)
        
        return features[:50]
    
    def _calculate_entropy(self, text: str) -> float:
        """Calculate Shannon entropy"""
        if not text:
            return 0
        
        char_counts = Counter(text)
        total = len(text)
        entropy = -sum((count/total) * np.log2(count/total) for count in char_counts.values())
        
        return entropy
    
    def _calculate_bigram_entropy(self, text: str) -> float:
        """Calculate bigram entropy"""
        if len(text) < 2:
            return 0
        
        bigrams = [text[i:i+2] for i in range(len(text)-1)]
        bigram_counts = Counter(bigrams)
        total = len(bigrams)
        
        entropy = -sum((count/total) * np.log2(count/total) for count in bigram_counts.values())
        
        return entropy
    
    def _calculate_entropy_from_counts(self, counts: Counter) -> float:
        """Calculate entropy from count dictionary"""
        total = sum(counts.values())
        if total == 0:
            return 0
        
        entropy = -sum((count/total) * np.log2(count/total) for count in counts.values())
        
        return entropy
    
    def _calculate_gini_coefficient(self, values: List[float]) -> float:
        """Calculate Gini coefficient"""
        sorted_values = sorted(values)
        n = len(sorted_values)
        
        if n == 0 or sum(sorted_values) == 0:
            return 0
        
        cumsum = np.cumsum(sorted_values)
        gini = (2 * np.sum((np.arange(n) + 1) * sorted_values)) / (n * cumsum[-1]) - (n + 1) / n
        
        return gini
    
    def _get_cache_key(self, column_name: str, values: List[Any]) -> str:
        """Generate cache key"""
        value_str = str(values[:5])
        key_str = f"{column_name}:{value_str}"
        return hashlib.md5(key_str.encode()).hexdigest()