#!/usr/bin/env python3

import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Set, Tuple, Any, Union
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
import hashlib
from scipy import stats
from scipy.spatial.distance import cosine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import logging
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class ColumnObservation:
    """Single observation of a column value"""
    value: Any
    table: str
    row_id: Optional[int] = None
    timestamp: Optional[float] = None
    
@dataclass  
class ColumnProfile:
    """Statistical profile of a column"""
    name: str
    observations: deque = field(default_factory=lambda: deque(maxlen=10000))
    frequency: int = 0
    tables: Set[str] = field(default_factory=set)
    
    # Statistical properties
    cardinality: float = 0.0
    null_ratio: float = 0.0
    unique_ratio: float = 0.0
    entropy: float = 0.0
    
    # Distribution properties
    length_distribution: Optional[np.ndarray] = None
    char_distribution: Optional[np.ndarray] = None
    pattern_distribution: Optional[Counter] = None
    
    # Learned embeddings
    embedding: Optional[np.ndarray] = None
    cluster_id: Optional[int] = None
    
class AdaptiveFeatureExtractor:
    """Adaptive feature extraction with online learning"""
    
    def __init__(self, dim: int = 768):
        self.dim = dim
        self.scaler = StandardScaler()
        self.pca = None
        self.feature_importance = None
        
        # Adaptive vocabulary for pattern learning
        self.pattern_vocab = {}
        self.vocab_size = 0
        
        # Character-level embeddings
        self.char_embeddings = self._init_char_embeddings()
        
        # Statistical moments tracker
        self.moment_tracker = defaultdict(lambda: {'n': 0, 'mean': 0, 'M2': 0, 'M3': 0, 'M4': 0})
        
    def _init_char_embeddings(self, dim: int = 64) -> np.ndarray:
        """Initialize character embeddings with linguistic priors"""
        embeddings = np.random.randn(256, dim) * 0.01
        
        # Similar characters get similar embeddings
        for i in range(ord('a'), ord('z')+1):
            embeddings[i] = embeddings[i-32] + np.random.randn(dim) * 0.001  # lowercase similar to uppercase
            
        for i in range(ord('0'), ord('9')+1):
            if i > ord('0'):
                embeddings[i] = embeddings[i-1] + np.random.randn(dim) * 0.001  # sequential numbers
                
        return embeddings
        
    def extract(self, column_name: str, observations: List[Any]) -> np.ndarray:
        """Extract comprehensive features using multiple representations"""
        
        features = []
        
        # 1. Name-based features (structural + semantic)
        name_features = self._extract_name_features(column_name)
        features.extend(name_features)
        
        # 2. Value-based features (statistical + distributional)
        if observations:
            value_features = self._extract_value_features(observations)
            features.extend(value_features)
        else:
            features.extend(np.zeros(384))
            
        # 3. Cross-features (name-value interaction)
        cross_features = self._extract_cross_features(column_name, observations)
        features.extend(cross_features)
        
        # 4. Higher-order features
        higher_order = self._extract_higher_order_features(np.array(features))
        features.extend(higher_order)
        
        features = np.array(features, dtype=np.float32)
        
        # Dimensionality reduction if needed
        if len(features) > self.dim:
            if self.pca is None:
                self.pca = PCA(n_components=self.dim, whiten=True)
                features = self.pca.fit_transform(features.reshape(1, -1))[0]
            else:
                features = self.pca.transform(features.reshape(1, -1))[0]
        elif len(features) < self.dim:
            features = np.pad(features, (0, self.dim - len(features)), mode='constant')
            
        return features
        
    def _extract_name_features(self, name: str) -> np.ndarray:
        """Extract rich features from column name"""
        features = []
        
        if not name:
            return np.zeros(256)
            
        name_lower = name.lower()
        
        # Character-level embedding
        char_emb = np.mean([self.char_embeddings[ord(c) if ord(c) < 256 else 0] for c in name], axis=0)
        features.extend(char_emb)
        
        # Morphological features
        tokens = re.split(r'[_\-\.\s]+', name_lower)
        features.append(len(tokens))
        features.append(np.mean([len(t) for t in tokens]))
        features.append(np.std([len(t) for t in tokens]))
        
        # Positional character statistics
        for pos in [0, -1]:  # First and last positions
            if abs(pos) <= len(name):
                char = name[pos]
                features.extend([
                    float(char.isalpha()),
                    float(char.isdigit()),
                    float(char.isupper()),
                    float(char in '_-.')
                ])
            else:
                features.extend([0, 0, 0, 0])
                
        # N-gram analysis
        for n in [2, 3, 4]:
            ngrams = [name[i:i+n] for i in range(len(name)-n+1)]
            if ngrams:
                # N-gram diversity
                features.append(len(set(ngrams)) / len(ngrams))
                # Repetition score
                features.append(max(Counter(ngrams).values()) / len(ngrams))
            else:
                features.extend([0, 0])
                
        # Suffix/prefix patterns
        common_prefixes = ['is_', 'has_', 'num_', 'total_', 'max_', 'min_']
        common_suffixes = ['_id', '_name', '_date', '_time', '_count', '_flag', '_type']
        
        features.extend([float(name_lower.startswith(p)) for p in common_prefixes])
        features.extend([float(name_lower.endswith(s)) for s in common_suffixes])
        
        # Information theory metrics
        if name:
            char_counts = Counter(name)
            probs = np.array(list(char_counts.values())) / len(name)
            entropy = -np.sum(probs * np.log2(probs + 1e-10))
            features.append(entropy)
        else:
            features.append(0)
            
        # Linguistic patterns
        vowels = set('aeiouAEIOU')
        consonants = set('bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ')
        
        vowel_ratio = sum(c in vowels for c in name) / max(len(name), 1)
        consonant_ratio = sum(c in consonants for c in name) / max(len(name), 1)
        features.extend([vowel_ratio, consonant_ratio])
        
        # Hash-based features for rare patterns
        hash_features = self._get_hash_features(name, num_features=32)
        features.extend(hash_features)
        
        return np.array(features[:256], dtype=np.float32)
        
    def _extract_value_features(self, observations: List[Any]) -> np.ndarray:
        """Extract sophisticated statistical features from values"""
        features = []
        
        if not observations:
            return np.zeros(384)
            
        # Convert to strings for analysis
        str_values = [str(v) if v is not None else '' for v in observations[:1000]]
        
        # Basic statistics
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
                stats.skew(lengths),
                stats.kurtosis(lengths)
            ])
        else:
            features.extend([0] * 9)
            
        # Cardinality and uniqueness
        unique_values = set(str_values)
        features.append(len(unique_values))
        features.append(len(unique_values) / max(len(str_values), 1))
        
        # Value pattern analysis
        patterns = self._analyze_value_patterns(str_values)
        features.extend(patterns)
        
        # Character-level statistics
        all_chars = ''.join(str_values)
        if all_chars:
            char_features = self._get_char_statistics(all_chars)
            features.extend(char_features)
        else:
            features.extend([0] * 20)
            
        # Temporal patterns (if values look like timestamps)
        temporal_features = self._extract_temporal_features(str_values)
        features.extend(temporal_features)
        
        # Numerical patterns (if values are numeric)
        numerical_features = self._extract_numerical_features(str_values)
        features.extend(numerical_features)
        
        # Structural patterns
        structural_features = self._extract_structural_features(str_values)
        features.extend(structural_features)
        
        # Distribution fitting
        dist_features = self._fit_distributions(str_values)
        features.extend(dist_features)
        
        return np.array(features[:384], dtype=np.float32)
        
    def _extract_cross_features(self, name: str, observations: List[Any]) -> np.ndarray:
        """Extract interaction features between name and values"""
        features = []
        
        if not observations:
            return np.zeros(128)
            
        name_lower = name.lower()
        str_values = [str(v) if v is not None else '' for v in observations[:100]]
        
        # Name-value consistency
        name_in_values = sum(name_lower in v.lower() for v in str_values) / max(len(str_values), 1)
        features.append(name_in_values)
        
        # Token overlap
        name_tokens = set(re.split(r'[_\-\.\s]+', name_lower))
        value_tokens = set()
        for v in str_values:
            value_tokens.update(re.split(r'[_\-\.\s]+', v.lower()))
            
        if name_tokens and value_tokens:
            jaccard = len(name_tokens & value_tokens) / len(name_tokens | value_tokens)
            features.append(jaccard)
        else:
            features.append(0)
            
        # Pattern correlation
        if '_id' in name_lower:
            has_numeric = sum(any(c.isdigit() for c in v) for v in str_values) / max(len(str_values), 1)
            features.append(has_numeric)
        else:
            features.append(0)
            
        # Length correlation
        avg_value_len = np.mean([len(v) for v in str_values]) if str_values else 0
        name_len = len(name)
        features.append(min(avg_value_len, name_len) / max(avg_value_len, name_len, 1))
        
        # Entropy correlation
        name_entropy = self._calculate_entropy(name)
        value_entropy = np.mean([self._calculate_entropy(v) for v in str_values[:20]]) if str_values else 0
        features.append(min(name_entropy, value_entropy) / max(name_entropy, value_entropy, 1))
        
        # Pad to fixed size
        return np.array(features[:128] + [0] * (128 - len(features)), dtype=np.float32)
        
    def _extract_higher_order_features(self, features: np.ndarray) -> np.ndarray:
        """Extract higher-order features through transformations"""
        higher_order = []
        
        # Polynomial features (selected interactions)
        if len(features) > 10:
            for i in range(min(10, len(features))):
                for j in range(i+1, min(10, len(features))):
                    higher_order.append(features[i] * features[j])
                    
        # Non-linear transformations
        higher_order.extend(np.tanh(features[:20]))
        higher_order.extend(np.log1p(np.abs(features[:20])))
        
        return np.array(higher_order[:64], dtype=np.float32)
        
    def _analyze_value_patterns(self, values: List[str]) -> List[float]:
        """Analyze patterns in values"""
        patterns = []
        
        # Pattern frequencies
        pattern_types = {
            'numeric': lambda v: v.isdigit(),
            'alpha': lambda v: v.isalpha(),
            'alphanumeric': lambda v: v.isalnum(),
            'has_special': lambda v: not v.isalnum() and v,
            'has_dots': lambda v: '.' in v,
            'has_dashes': lambda v: '-' in v,
            'has_underscores': lambda v: '_' in v,
            'has_spaces': lambda v: ' ' in v,
            'has_at': lambda v: '@' in v,
            'has_slash': lambda v: '/' in v,
            'starts_digit': lambda v: v and v[0].isdigit(),
            'ends_digit': lambda v: v and v[-1].isdigit(),
            'mixed_case': lambda v: v != v.lower() and v != v.upper(),
            'ip_like': lambda v: bool(re.match(r'^[\d\.]+$', v)),
            'uuid_like': lambda v: bool(re.match(r'^[a-f0-9\-]+$', v, re.I)) and len(v) > 20,
            'email_like': lambda v: '@' in v and '.' in v,
            'url_like': lambda v: v.startswith(('http://', 'https://', 'www.')),
            'path_like': lambda v: '/' in v and len(v.split('/')) > 2,
        }
        
        for pattern_name, pattern_func in pattern_types.items():
            ratio = sum(pattern_func(v) for v in values) / max(len(values), 1)
            patterns.append(ratio)
            
        return patterns
        
    def _get_char_statistics(self, text: str) -> List[float]:
        """Get character-level statistics"""
        stats = []
        
        total = len(text)
        if total == 0:
            return [0] * 20
            
        # Character type ratios
        stats.append(sum(c.isalpha() for c in text) / total)
        stats.append(sum(c.isdigit() for c in text) / total)
        stats.append(sum(c.isspace() for c in text) / total)
        stats.append(sum(c.isupper() for c in text) / total)
        stats.append(sum(c.islower() for c in text) / total)
        
        # Special character ratios
        for chars in ['.,-_', '!@#$%', '()[]{}', '/:;', '"\'`']:
            stats.append(sum(c in chars for c in text) / total)
            
        # Unicode categories
        stats.append(sum(ord(c) > 127 for c in text) / total)  # Non-ASCII
        
        # Consecutive patterns
        consecutive_digits = len(re.findall(r'\d{2,}', text))
        consecutive_alpha = len(re.findall(r'[a-zA-Z]{3,}', text))
        stats.extend([consecutive_digits / total, consecutive_alpha / total])
        
        # Character diversity
        unique_chars = len(set(text))
        stats.append(unique_chars / min(total, 256))
        
        return stats
        
    def _extract_temporal_features(self, values: List[str]) -> List[float]:
        """Extract features if values might be temporal"""
        features = []
        
        # Common date/time patterns
        patterns = [
            (r'\d{4}-\d{2}-\d{2}', 'iso_date'),
            (r'\d{2}/\d{2}/\d{4}', 'us_date'),
            (r'\d{10,13}', 'timestamp'),
            (r'\d{2}:\d{2}:\d{2}', 'time'),
        ]
        
        for pattern, name in patterns:
            matches = sum(bool(re.search(pattern, v)) for v in values)
            features.append(matches / max(len(values), 1))
            
        return features
        
    def _extract_numerical_features(self, values: List[str]) -> List[float]:
        """Extract features for numerical values"""
        features = []
        
        # Try to parse as numbers
        numbers = []
        for v in values:
            try:
                num = float(v)
                numbers.append(num)
            except:
                pass
                
        if numbers:
            features.extend([
                np.mean(numbers),
                np.std(numbers),
                np.min(numbers),
                np.max(numbers),
                len(numbers) / len(values),  # Parse success ratio
            ])
            
            # Check for integer vs float
            int_ratio = sum(n == int(n) for n in numbers) / len(numbers)
            features.append(int_ratio)
        else:
            features.extend([0] * 6)
            
        return features
        
    def _extract_structural_features(self, values: List[str]) -> List[float]:
        """Extract structural features from values"""
        features = []
        
        # Delimiter analysis
        delimiters = ['.', '-', '_', '/', ':', ',', ';', '|']
        for delim in delimiters:
            count = sum(delim in v for v in values) / max(len(values), 1)
            features.append(count)
            
            # Average number of parts when split
            if count > 0:
                parts = [len(v.split(delim)) for v in values if delim in v]
                features.append(np.mean(parts) if parts else 0)
            else:
                features.append(0)
                
        # Hierarchical structure (dots suggest hierarchy)
        dot_depths = [v.count('.') for v in values]
        features.append(np.mean(dot_depths) if dot_depths else 0)
        features.append(np.std(dot_depths) if dot_depths else 0)
        
        return features
        
    def _fit_distributions(self, values: List[str]) -> List[float]:
        """Fit statistical distributions to value lengths"""
        features = []
        
        lengths = [len(v) for v in values]
        if not lengths or len(set(lengths)) == 1:
            return [0] * 10
            
        # Fit normal distribution
        mu, sigma = np.mean(lengths), np.std(lengths)
        features.extend([mu, sigma])
        
        # Fit exponential
        if min(lengths) > 0:
            lambda_param = 1 / np.mean(lengths)
            features.append(lambda_param)
        else:
            features.append(0)
            
        # Fit power law (simplified)
        if min(lengths) > 0:
            log_lengths = np.log(lengths)
            alpha = 1 + len(lengths) / np.sum(log_lengths - np.min(log_lengths))
            features.append(alpha)
        else:
            features.append(0)
            
        # KS test for normality
        if len(lengths) > 10:
            _, p_value = stats.kstest(lengths, 'norm', args=(mu, sigma))
            features.append(p_value)
        else:
            features.append(0.5)
            
        # Entropy of length distribution
        length_counts = Counter(lengths)
        probs = np.array(list(length_counts.values())) / len(lengths)
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        features.append(entropy)
        
        # Quantiles
        features.extend(np.percentile(lengths, [10, 25, 50, 75, 90]))
        
        return features[:10]
        
    def _calculate_entropy(self, text: str) -> float:
        """Calculate Shannon entropy of text"""
        if not text:
            return 0
            
        char_counts = Counter(text)
        probs = np.array(list(char_counts.values())) / len(text)
        return -np.sum(probs * np.log2(probs + 1e-10))
        
    def _get_hash_features(self, text: str, num_features: int = 32) -> List[float]:
        """Generate hash-based features for rare patterns"""
        features = [0] * num_features
        
        # Use multiple hash functions
        for i, seed in enumerate([42, 137, 256]):
            hash_val = int(hashlib.md5(f"{seed}{text}".encode()).hexdigest(), 16)
            features[hash_val % num_features] = 1.0
            
        return features
        
    def update_online(self, features: np.ndarray):
        """Online update of feature statistics"""
        # Update running statistics
        n = self.moment_tracker['global']['n']
        self.moment_tracker['global']['n'] += 1
        
        # Welford's online algorithm for mean and variance
        delta = features - self.moment_tracker['global']['mean']
        self.moment_tracker['global']['mean'] += delta / (n + 1)
        
        delta2 = features - self.moment_tracker['global']['mean']
        self.moment_tracker['global']['M2'] += delta * delta2

class HierarchicalAttentionNetwork(nn.Module):
    """Hierarchical attention network for column type classification"""
    
    def __init__(self, input_dim: int = 768, hidden_dim: int = 384, 
                 num_heads: int = 12, num_types: int = 100, device: str = 'mps'):
        super().__init__()
        
        self.device = device
        
        # Multi-scale feature extractors
        self.local_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.global_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # Hierarchical classifier
        self.type_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_types)
        )
        
        # Confidence estimator
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.to(device)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with confidence estimation"""
        
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
            
        x = x.to(self.device)
        
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
            
        # Multi-scale encoding
        local_features = self.local_encoder(x)
        global_features = self.global_encoder(x)
        
        # Self-attention
        attended, attention_weights = self.attention(
            local_features.unsqueeze(1),
            local_features.unsqueeze(1),
            local_features.unsqueeze(1)
        )
        attended = attended.squeeze(1)
        
        # Combine features
        combined = torch.cat([attended, global_features], dim=-1)
        
        # Classification
        logits = self.type_classifier(combined)
        confidence = self.confidence_head(combined)
        
        return logits, confidence

class ColumnClassifier:
    """Main column classifier with zero assumptions"""
    
    def __init__(self, device: str = 'mps'):
        self.device = device
        
        # Column profiles
        self.profiles = {}  # column_name -> ColumnProfile
        
        # Feature extractor
        self.feature_extractor = AdaptiveFeatureExtractor(dim=768)
        
        # Clustering for type discovery
        self.clusterer = None
        self.cluster_centers = []
        self.type_names = {}  # cluster_id -> semantic name
        
        # Neural classifier (initialized after discovery)
        self.classifier = None
        
        # Statistics
        self.stats = defaultdict(int)
        
    def observe(self, column_name: str, value: Any, table: str):
        """Observe a column value during scanning"""
        
        if column_name not in self.profiles:
            self.profiles[column_name] = ColumnProfile(name=column_name)
            
        profile = self.profiles[column_name]
        profile.observations.append(ColumnObservation(value, table))
        profile.frequency += 1
        profile.tables.add(table)
        
        self.stats['total_observations'] += 1
        
    def discover_types(self):
        """Discover column types from observations using unsupervised learning"""
        
        logger.info(f"Discovering types from {len(self.profiles)} columns")
        
        # Extract features for all columns
        features = []
        column_names = []
        
        for name, profile in tqdm(self.profiles.items(), desc="Extracting features"):
            if profile.frequency < 10:  # Skip rare columns
                continue
                
            # Get sample values
            sample_values = [obs.value for obs in profile.observations]
            
            # Extract features
            feature_vec = self.feature_extractor.extract(name, sample_values)
            features.append(feature_vec)
            column_names.append(name)
            
            # Store embedding
            profile.embedding = feature_vec
            
            # Calculate statistics
            self._calculate_profile_statistics(profile)
            
        features = np.array(features)
        
        # Normalize features
        features = StandardScaler().fit_transform(features)
        
        # Dimensionality reduction for clustering
        if features.shape[0] > 50:
            pca = PCA(n_components=min(50, features.shape[0]))
            features_reduced = pca.fit_transform(features)
        else:
            features_reduced = features
            
        # Adaptive clustering with DBSCAN
        self.clusterer = DBSCAN(
            eps=0.5,
            min_samples=max(2, len(features) // 100),
            metric='cosine',
            n_jobs=-1
        )
        
        clusters = self.clusterer.fit_predict(features_reduced)
        
        # Store cluster assignments
        for i, (name, cluster_id) in enumerate(zip(column_names, clusters)):
            self.profiles[name].cluster_id = cluster_id
            
        # Find cluster centers and name them
        unique_clusters = set(clusters) - {-1}  # Exclude noise
        
        logger.info(f"Discovered {len(unique_clusters)} column types")
        
        for cluster_id in unique_clusters:
            cluster_indices = [i for i, c in enumerate(clusters) if c == cluster_id]
            cluster_features = features[cluster_indices]
            
            # Calculate center
            center = np.mean(cluster_features, axis=0)
            self.cluster_centers.append(center)
            
            # Semantic naming based on column names in cluster
            cluster_columns = [column_names[i] for i in cluster_indices]
            self.type_names[cluster_id] = self._infer_type_name(cluster_columns)
            
            logger.info(f"Type {cluster_id}: {self.type_names[cluster_id]} ({len(cluster_columns)} columns)")
            
        # Build neural classifier
        self._build_classifier(features, clusters)
        
    def _calculate_profile_statistics(self, profile: ColumnProfile):
        """Calculate statistical properties of a column"""
        
        values = [obs.value for obs in profile.observations]
        
        # Cardinality
        unique_values = set(values)
        profile.cardinality = len(unique_values)
        profile.unique_ratio = profile.cardinality / max(len(values), 1)
        
        # Null ratio
        null_count = sum(v is None or v == '' for v in values)
        profile.null_ratio = null_count / max(len(values), 1)
        
        # Entropy
        if values:
            value_counts = Counter(values)
            probs = np.array(list(value_counts.values())) / len(values)
            profile.entropy = -np.sum(probs * np.log2(probs + 1e-10))
            
        # Length distribution for strings
        str_values = [str(v) if v is not None else '' for v in values]
        lengths = [len(v) for v in str_values]
        profile.length_distribution = np.array(lengths)
        
    def _infer_type_name(self, column_names: List[str]) -> str:
        """Infer semantic type name from column names in cluster"""
        
        # Find common tokens
        all_tokens = []
        for name in column_names:
            tokens = re.split(r'[_\-\.\s]+', name.lower())
            all_tokens.extend(tokens)
            
        # Most common tokens
        token_counts = Counter(all_tokens)
        common_tokens = [token for token, count in token_counts.most_common(3) 
                        if count > len(column_names) * 0.3]
        
        if common_tokens:
            return f"type_{'_'.join(common_tokens)}"
        else:
            return f"type_cluster_{len(self.type_names)}"
            
    def _build_classifier(self, features: np.ndarray, labels: np.ndarray):
        """Build and train neural classifier"""
        
        # Filter out noise points
        valid_indices = labels != -1
        features = features[valid_indices]
        labels = labels[valid_indices]
        
        if len(set(labels)) < 2:
            logger.warning("Not enough types for classification")
            return
            
        # Initialize classifier
        num_types = len(set(labels))
        self.classifier = HierarchicalAttentionNetwork(
            input_dim=features.shape[1],
            num_types=num_types,
            device=self.device
        )
        
        # Prepare training data
        dataset = torch.utils.data.TensorDataset(
            torch.tensor(features, dtype=torch.float32),
            torch.tensor(labels, dtype=torch.long)
        )
        
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=min(32, len(dataset)),
            shuffle=True
        )
        
        # Training
        optimizer = torch.optim.AdamW(self.classifier.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        criterion = nn.CrossEntropyLoss()
        
        self.classifier.train()
        
        for epoch in range(100):
            total_loss = 0
            
            for batch_features, batch_labels in dataloader:
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                optimizer.zero_grad()
                
                logits, confidence = self.classifier(batch_features)
                loss = criterion(logits, batch_labels)
                
                # Confidence regularization
                conf_loss = -torch.mean(torch.log(confidence + 1e-10))
                total_loss_batch = loss + 0.1 * conf_loss
                
                total_loss_batch.backward()
                torch.nn.utils.clip_grad_norm_(self.classifier.parameters(), 1.0)
                optimizer.step()
                
                total_loss += total_loss_batch.item()
                
            scheduler.step()
            
            if epoch % 20 == 0:
                logger.info(f"Epoch {epoch}, Loss: {total_loss:.4f}")
                
        self.classifier.eval()
        
    def classify(self, column_name: str, sample_values: List[Any] = None) -> Dict[str, Any]:
        """Classify a column based on learned types"""
        
        # Extract features
        if column_name in self.profiles:
            profile = self.profiles[column_name]
            features = profile.embedding
            
            if features is None:
                sample_values = [obs.value for obs in profile.observations]
                features = self.feature_extractor.extract(column_name, sample_values)
        else:
            features = self.feature_extractor.extract(column_name, sample_values or [])
            
        # Classify
        if self.classifier:
            with torch.no_grad():
                logits, confidence = self.classifier(torch.tensor(features, dtype=torch.float32))
                
                # Get top predictions
                probs = F.softmax(logits, dim=-1)
                top_k = min(3, probs.shape[-1])
                top_probs, top_indices = torch.topk(probs, top_k)
                
                predictions = []
                for prob, idx in zip(top_probs[0], top_indices[0]):
                    cluster_id = idx.item()
                    predictions.append({
                        'type': self.type_names.get(cluster_id, f'type_{cluster_id}'),
                        'probability': prob.item(),
                        'confidence': confidence[0].item()
                    })
                    
                return {
                    'predictions': predictions,
                    'primary_type': predictions[0]['type'] if predictions else 'unknown',
                    'confidence': predictions[0]['confidence'] if predictions else 0.0
                }
        else:
            # Fallback to nearest neighbor
            if self.cluster_centers:
                distances = [cosine(features, center) for center in self.cluster_centers]
                nearest = np.argmin(distances)
                
                return {
                    'primary_type': self.type_names.get(nearest, f'type_{nearest}'),
                    'confidence': 1.0 / (1.0 + distances[nearest])
                }
                
        return {'primary_type': 'unknown', 'confidence': 0.0}
        
    def get_important_columns(self, top_n: int = 100) -> List[Tuple[str, float]]:
        """Get most important columns based on multiple factors"""
        
        importance_scores = []
        
        for name, profile in self.profiles.items():
            # Multi-factor importance
            frequency_score = np.log1p(profile.frequency)
            table_score = np.log1p(len(profile.tables))
            cardinality_score = np.log1p(profile.cardinality)
            entropy_score = profile.entropy if profile.entropy else 0
            
            # Weighted combination
            importance = (
                0.3 * frequency_score +
                0.2 * table_score +
                0.3 * cardinality_score +
                0.2 * entropy_score
            )
            
            importance_scores.append((name, importance))
            
        # Sort by importance
        importance_scores.sort(key=lambda x: x[1], reverse=True)
        
        return importance_scores[:top_n]