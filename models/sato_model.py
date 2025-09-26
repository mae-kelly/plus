import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class SimpleLDA:
    """Simple LDA implementation without sklearn"""
    def __init__(self, n_components: int = 400):
        self.n_components = n_components
        self.vocabulary = {}
        self.topic_word_dist = None
        self.fitted = False
        
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """Simple topic modeling without sklearn"""
        if not texts:
            return np.zeros((1, self.n_components))
        
        # Build vocabulary from texts
        from collections import Counter
        word_counts = Counter()
        for text in texts:
            words = text.lower().split()
            word_counts.update(words)
        
        # Keep top words
        vocab_items = word_counts.most_common(min(5000, len(word_counts)))
        self.vocabulary = {word: idx for idx, (word, _) in enumerate(vocab_items)}
        
        # Create document-term matrix
        n_docs = len(texts)
        n_words = len(self.vocabulary)
        doc_term_matrix = np.zeros((n_docs, n_words))
        
        for doc_idx, text in enumerate(texts):
            words = text.lower().split()
            for word in words:
                if word in self.vocabulary:
                    word_idx = self.vocabulary[word]
                    doc_term_matrix[doc_idx, word_idx] += 1
        
        # Simple topic assignment (random projection for simplicity)
        np.random.seed(42)
        self.topic_word_dist = np.random.dirichlet(np.ones(n_words), self.n_components)
        
        # Document-topic distribution
        doc_topic_dist = np.zeros((n_docs, self.n_components))
        for doc_idx in range(n_docs):
            for topic_idx in range(self.n_components):
                doc_topic_dist[doc_idx, topic_idx] = np.dot(
                    doc_term_matrix[doc_idx], 
                    self.topic_word_dist[topic_idx]
                )
        
        # Normalize
        doc_topic_dist = doc_topic_dist / (doc_topic_dist.sum(axis=1, keepdims=True) + 1e-10)
        
        self.fitted = True
        return doc_topic_dist
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """Transform new texts to topic distribution"""
        if not self.fitted:
            return self.fit_transform(texts)
        
        n_docs = len(texts)
        n_words = len(self.vocabulary)
        
        # Create document-term matrix for new texts
        doc_term_matrix = np.zeros((n_docs, n_words))
        for doc_idx, text in enumerate(texts):
            words = text.lower().split()
            for word in words:
                if word in self.vocabulary:
                    word_idx = self.vocabulary[word]
                    doc_term_matrix[doc_idx, word_idx] += 1
        
        # Calculate document-topic distribution
        doc_topic_dist = np.zeros((n_docs, self.n_components))
        for doc_idx in range(n_docs):
            for topic_idx in range(self.n_components):
                doc_topic_dist[doc_idx, topic_idx] = np.dot(
                    doc_term_matrix[doc_idx],
                    self.topic_word_dist[topic_idx]
                )
        
        # Normalize
        doc_topic_dist = doc_topic_dist / (doc_topic_dist.sum(axis=1, keepdims=True) + 1e-10)
        
        return doc_topic_dist

class SimpleVectorizer:
    """Simple count vectorizer without sklearn"""
    def __init__(self, max_features: int = 5000):
        self.max_features = max_features
        self.vocabulary = {}
        
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """Fit and transform texts to count matrix"""
        from collections import Counter
        
        # Count all words
        word_counts = Counter()
        for text in texts:
            words = text.lower().split()
            word_counts.update(words)
        
        # Build vocabulary with most common words
        vocab_items = word_counts.most_common(min(self.max_features, len(word_counts)))
        self.vocabulary = {word: idx for idx, (word, _) in enumerate(vocab_items)}
        
        # Create count matrix
        n_docs = len(texts)
        n_features = len(self.vocabulary)
        matrix = np.zeros((n_docs, n_features))
        
        for doc_idx, text in enumerate(texts):
            words = text.lower().split()
            for word in words:
                if word in self.vocabulary:
                    matrix[doc_idx, self.vocabulary[word]] += 1
        
        return matrix

class SatoModel(nn.Module):
    def __init__(self, num_topics: int = 400, num_types: int = 78, device: str = 'mps'):
        super().__init__()
        self.device = device
        self.num_topics = num_topics
        
        # Use simple LDA instead of sklearn
        self.lda = SimpleLDA(n_components=num_topics)
        self.vectorizer = SimpleVectorizer(max_features=5000)
        
        # Topic encoder
        self.topic_encoder = nn.Sequential(
            nn.Linear(num_topics, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Context encoder
        self.context_encoder = nn.LSTM(
            input_size=128,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )
        
        # CRF layer
        self.crf_layer = CRF(num_types, batch_first=True)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_types)
        )
        
        self.to(device)
    
    def forward(self, topic_features, context_features=None):
        topic_encoded = self.topic_encoder(topic_features)
        
        if context_features is not None:
            context_out, _ = self.context_encoder(context_features.unsqueeze(1))
            context_out = context_out.squeeze(1)
            combined = torch.cat([topic_encoded, context_out], dim=-1)
        else:
            combined = torch.cat([topic_encoded, torch.zeros_like(topic_encoded)], dim=-1)
        
        logits = self.classifier(combined)
        return logits
    
    def extract_topic_features(self, values: List[str]) -> np.ndarray:
        """Extract topic features from values"""
        if not values or not any(values):
            return np.zeros(self.num_topics)
        
        try:
            text_values = [str(v) for v in values if v]
            
            # Fit the model if not fitted
            if not self.lda.fitted and len(text_values) > 10:
                corpus = text_values[:1000]
                self.lda.fit_transform(corpus)
            
            if self.lda.fitted:
                topic_dist = self.lda.transform(text_values)
                return np.mean(topic_dist, axis=0)
            else:
                return np.zeros(self.num_topics)
            
        except Exception as e:
            logger.debug(f"Topic extraction failed: {e}")
            return np.zeros(self.num_topics)
    
    def extract_context_features(self, table_data: Dict[str, List]) -> np.ndarray:
        """Extract context features from entire table"""
        context_features = []
        
        for column_name, values in table_data.items():
            column_topic = self.extract_topic_features(values[:100])
            context_features.append(column_topic)
        
        if context_features:
            return np.mean(context_features, axis=0)
        else:
            return np.zeros(self.num_topics)

class CRF(nn.Module):
    """Conditional Random Field layer"""
    def __init__(self, num_tags, batch_first=True):
        super().__init__()
        self.num_tags = num_tags
        self.batch_first = batch_first
        
        self.start_transitions = nn.Parameter(torch.randn(num_tags))
        self.end_transitions = nn.Parameter(torch.randn(num_tags))
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))
    
    def forward(self, emissions, mask=None):
        if mask is None:
            mask = torch.ones(emissions.shape[:2], dtype=torch.bool)
        
        score = self._compute_score(emissions, mask)
        partition = self._compute_partition(emissions, mask)
        
        return partition - score
    
    def _compute_score(self, emissions, mask):
        batch_size, seq_len = mask.shape
        score = self.start_transitions[None, :] + emissions[:, 0]
        
        for i in range(1, seq_len):
            broadcast_score = score.unsqueeze(2)
            broadcast_emissions = emissions[:, i].unsqueeze(1)
            
            next_score = broadcast_score + self.transitions + broadcast_emissions
            next_score = torch.logsumexp(next_score, dim=1)
            
            score = torch.where(mask[:, i].unsqueeze(1), next_score, score)
        
        score = score + self.end_transitions[None, :]
        score = torch.logsumexp(score, dim=1)
        
        return score
    
    def _compute_partition(self, emissions, mask):
        batch_size = emissions.shape[0]
        
        alpha = self.start_transitions[None, :] + emissions[:, 0]
        
        for i in range(1, emissions.shape[1]):
            broadcast_alpha = alpha.unsqueeze(2)
            broadcast_emissions = emissions[:, i].unsqueeze(1)
            
            inner = broadcast_alpha + self.transitions + broadcast_emissions
            alpha = torch.logsumexp(inner, dim=1)
            
            alpha = torch.where(mask[:, i].unsqueeze(1), alpha, alpha)
        
        alpha = alpha + self.end_transitions[None, :]
        
        return torch.logsumexp(alpha, dim=1)