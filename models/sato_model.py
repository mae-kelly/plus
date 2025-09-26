import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any
from collections import Counter, defaultdict
import re
import logging

logger = logging.getLogger(__name__)

class LocalLDA:
    """Local implementation of topic modeling without sklearn"""
    def __init__(self, n_components: int = 400, max_iterations: int = 100):
        self.n_components = n_components
        self.max_iterations = max_iterations
        self.topics = None
        self.vocabulary = {}
        self.doc_topic_dist = None
        
    def fit_transform(self, documents: List[str]) -> np.ndarray:
        """Fit the model and transform documents to topic distributions"""
        # Build vocabulary
        self._build_vocabulary(documents)
        
        # Create document-term matrix
        doc_term_matrix = self._create_doc_term_matrix(documents)
        
        # Initialize topics randomly
        n_docs, n_terms = doc_term_matrix.shape
        self.doc_topic_dist = np.random.dirichlet(np.ones(self.n_components), n_docs)
        self.topic_term_dist = np.random.dirichlet(np.ones(n_terms), self.n_components)
        
        # Simple iterative update (simplified LDA)
        for _ in range(self.max_iterations):
            # Update document-topic distribution
            for d in range(n_docs):
                for t in range(self.n_components):
                    self.doc_topic_dist[d, t] = np.sum(
                        doc_term_matrix[d] * self.topic_term_dist[t]
                    )
            
            # Normalize
            self.doc_topic_dist = self.doc_topic_dist / self.doc_topic_dist.sum(axis=1, keepdims=True)
        
        return self.doc_topic_dist
    
    def transform(self, documents: List[str]) -> np.ndarray:
        """Transform documents to topic distributions"""
        doc_term_matrix = self._create_doc_term_matrix(documents)
        n_docs = doc_term_matrix.shape[0]
        
        doc_topics = np.zeros((n_docs, self.n_components))
        
        for d in range(n_docs):
            for t in range(self.n_components):
                doc_topics[d, t] = np.sum(
                    doc_term_matrix[d] * self.topic_term_dist[t]
                )
        
        # Normalize
        doc_topics = doc_topics / (doc_topics.sum(axis=1, keepdims=True) + 1e-10)
        
        return doc_topics
    
    def _build_vocabulary(self, documents: List[str]):
        """Build vocabulary from documents"""
        word_counts = Counter()
        
        for doc in documents:
            words = re.findall(r'\w+', doc.lower())
            word_counts.update(words)
        
        # Keep top 5000 words
        most_common = word_counts.most_common(5000)
        self.vocabulary = {word: idx for idx, (word, _) in enumerate(most_common)}
    
    def _create_doc_term_matrix(self, documents: List[str]) -> np.ndarray:
        """Create document-term matrix"""
        n_docs = len(documents)
        n_terms = len(self.vocabulary)
        
        matrix = np.zeros((n_docs, n_terms))
        
        for d, doc in enumerate(documents):
            words = re.findall(r'\w+', doc.lower())
            for word in words:
                if word in self.vocabulary:
                    matrix[d, self.vocabulary[word]] += 1
        
        return matrix

class SatoModel(nn.Module):
    def __init__(self, num_topics: int = 400, num_types: int = 78, device: str = 'mps'):
        super().__init__()
        self.device = device
        self.num_topics = num_topics
        
        # Local topic model
        self.lda = LocalLDA(n_components=num_topics)
        self.fitted = False
        
        # Topic encoder
        self.topic_encoder = nn.Sequential(
            nn.Linear(num_topics, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Context encoder using LSTM
        self.context_encoder = nn.LSTM(
            input_size=128,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(512, 8, dropout=0.1, batch_first=True)
        
        # CRF layer for structured prediction
        self.crf_layer = CRF(num_types, batch_first=True)
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_types)
        )
        
        # Additional pattern-based features
        self.pattern_encoder = nn.Sequential(
            nn.Linear(50, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64)
        )
        
        # Combined classifier
        self.combined_classifier = nn.Sequential(
            nn.Linear(512 + 64, 384),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(384, num_types)
        )
        
        self.to(device)
    
    def forward(self, topic_features, context_features=None, pattern_features=None):
        # Encode topics
        topic_encoded = self.topic_encoder(topic_features)
        
        # Process context if available
        if context_features is not None:
            context_out, _ = self.context_encoder(context_features.unsqueeze(1))
            context_out = context_out.squeeze(1)
            
            # Apply attention
            combined = torch.cat([topic_encoded.unsqueeze(1), context_out.unsqueeze(1)], dim=2)
            attended, _ = self.attention(combined, combined, combined)
            combined = attended.squeeze(1)
        else:
            # Duplicate topic encoding to match expected dimension
            combined = torch.cat([topic_encoded, topic_encoded, 
                                 torch.zeros_like(topic_encoded), torch.zeros_like(topic_encoded)], dim=-1)
        
        # Add pattern features if available
        if pattern_features is not None:
            pattern_encoded = self.pattern_encoder(pattern_features)
            logits = self.combined_classifier(torch.cat([combined, pattern_encoded], dim=-1))
        else:
            logits = self.classifier(combined)
        
        return logits
    
    def extract_topic_features(self, values: List[str]) -> np.ndarray:
        """Extract topic features from values"""
        if not values or not any(values):
            return np.zeros(self.num_topics)
        
        try:
            text_values = [str(v) for v in values if v]
            
            # Fit the model if not fitted
            if not self.fitted and len(text_values) > 10:
                corpus = text_values[:1000]
                self.lda.fit_transform(corpus)
                self.fitted = True
            
            # Transform to topic distribution
            if self.fitted:
                topic_dist = self.lda.transform(text_values)
                return np.mean(topic_dist, axis=0)
            else:
                # Return random distribution if not fitted
                return np.random.dirichlet(np.ones(self.num_topics))
            
        except Exception as e:
            logger.debug(f"Topic extraction failed: {e}")
            return np.zeros(self.num_topics)
    
    def extract_context_features(self, table_data: Dict[str, List]) -> np.ndarray:
        """Extract context features from entire table"""
        context_features = []
        
        for column_name, values in table_data.items():
            # Extract topic features for each column
            column_topic = self.extract_topic_features(values[:100])
            context_features.append(column_topic)
        
        if context_features:
            return np.mean(context_features, axis=0)
        else:
            return np.zeros(self.num_topics)
    
    def extract_pattern_features(self, values: List[str]) -> np.ndarray:
        """Extract pattern-based features"""
        features = []
        
        str_values = [str(v) if v is not None else '' for v in values[:100]]
        
        if not str_values:
            return np.zeros(50)
        
        # Length statistics
        lengths = [len(v) for v in str_values]
        features.extend([
            np.mean(lengths) if lengths else 0,
            np.std(lengths) if len(lengths) > 1 else 0,
            np.min(lengths) if lengths else 0,
            np.max(lengths) if lengths else 0,
        ])
        
        # Character type ratios
        all_chars = ''.join(str_values)
        total_chars = max(len(all_chars), 1)
        
        features.extend([
            sum(c.isalpha() for c in all_chars) / total_chars,
            sum(c.isdigit() for c in all_chars) / total_chars,
            sum(c.isspace() for c in all_chars) / total_chars,
            sum(c in '.-_@' for c in all_chars) / total_chars,
        ])
        
        # Pattern detection
        patterns = {
            'email': r'@.*\.',
            'url': r'https?://',
            'ip': r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}',
            'date': r'\d{4}-\d{2}-\d{2}',
            'phone': r'\d{3}[-.\s]?\d{3}[-.\s]?\d{4}',
            'uuid': r'[a-f0-9]{8}-[a-f0-9]{4}',
        }
        
        for pattern_name, pattern_regex in patterns.items():
            matches = sum(1 for v in str_values if re.search(pattern_regex, v, re.I))
            features.append(matches / len(str_values))
        
        # Uniqueness and entropy
        unique_ratio = len(set(str_values)) / len(str_values)
        features.append(unique_ratio)
        
        # Value distribution
        value_counts = Counter(str_values)
        frequencies = list(value_counts.values())
        
        if frequencies:
            features.extend([
                max(frequencies) / len(str_values),
                min(frequencies) / len(str_values),
                np.mean(frequencies) / len(str_values),
            ])
        else:
            features.extend([0, 0, 0])
        
        # Pad to fixed size
        while len(features) < 50:
            features.append(0)
        
        return np.array(features[:50], dtype=np.float32)

class CRF(nn.Module):
    """Conditional Random Field layer for structured prediction"""
    def __init__(self, num_tags, batch_first=True):
        super().__init__()
        self.num_tags = num_tags
        self.batch_first = batch_first
        
        # Transition parameters
        self.start_transitions = nn.Parameter(torch.randn(num_tags))
        self.end_transitions = nn.Parameter(torch.randn(num_tags))
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))
    
    def forward(self, emissions, mask=None):
        """Forward pass of CRF"""
        if mask is None:
            batch_size, seq_len = emissions.shape[:2]
            mask = torch.ones((batch_size, seq_len), dtype=torch.bool, device=emissions.device)
        
        # Compute score and partition
        score = self._compute_score(emissions, mask)
        partition = self._compute_partition(emissions, mask)
        
        return partition - score
    
    def _compute_score(self, emissions, mask):
        """Compute score of a sequence"""
        batch_size, seq_len = mask.shape
        score = self.start_transitions[None, :] + emissions[:, 0]
        
        for i in range(1, seq_len):
            broadcast_score = score.unsqueeze(2)
            broadcast_emissions = emissions[:, i].unsqueeze(1)
            
            next_score = broadcast_score + self.transitions + broadcast_emissions
            next_score = torch.logsumexp(next_score, dim=1)
            
            score = torch.where(mask[:, i].unsqueeze(1), next_score, score)
        
        # Add end transitions
        score = score + self.end_transitions[None, :]
        score = torch.logsumexp(score, dim=1)
        
        return score
    
    def _compute_partition(self, emissions, mask):
        """Compute partition function"""
        batch_size = emissions.shape[0]
        
        # Initialize alpha
        alpha = self.start_transitions[None, :] + emissions[:, 0]
        
        for i in range(1, emissions.shape[1]):
            broadcast_alpha = alpha.unsqueeze(2)
            broadcast_emissions = emissions[:, i].unsqueeze(1)
            
            inner = broadcast_alpha + self.transitions + broadcast_emissions
            alpha = torch.logsumexp(inner, dim=1)
            
            alpha = torch.where(mask[:, i].unsqueeze(1), alpha, alpha)
        
        # Add end transitions
        alpha = alpha + self.end_transitions[None, :]
        
        return torch.logsumexp(alpha, dim=1)
    
    def viterbi_decode(self, emissions, mask=None):
        """Viterbi decoding to find the best sequence"""
        if mask is None:
            batch_size, seq_len = emissions.shape[:2]
            mask = torch.ones((batch_size, seq_len), dtype=torch.bool, device=emissions.device)
        
        batch_size, seq_len, num_tags = emissions.shape
        
        # Initialize scores
        scores = self.start_transitions[None, :] + emissions[:, 0]
        paths = []
        
        for i in range(1, seq_len):
            broadcast_scores = scores.unsqueeze(2)
            broadcast_emissions = emissions[:, i].unsqueeze(1)
            
            scores_with_trans = broadcast_scores + self.transitions + broadcast_emissions
            
            scores, indices = torch.max(scores_with_trans, dim=1)
            paths.append(indices)
        
        # Add end transitions and find best path
        scores = scores + self.end_transitions[None, :]
        _, best_last_tag = torch.max(scores, dim=1)
        
        # Backtrack
        best_paths = [best_last_tag]
        for indices in reversed(paths):
            best_last_tag = torch.gather(indices, 1, best_last_tag.unsqueeze(1)).squeeze(1)
            best_paths.append(best_last_tag)
        
        best_paths.reverse()
        
        return torch.stack(best_paths, dim=1)