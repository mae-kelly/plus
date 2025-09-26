import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class SatoModel(nn.Module):
    def __init__(self, num_topics: int = 400, num_types: int = 78, device: str = 'mps'):
        super().__init__()
        self.device = device
        self.num_topics = num_topics
        
        self.lda = LatentDirichletAllocation(
            n_components=num_topics,
            learning_method='batch',
            random_state=42
        )
        self.vectorizer = CountVectorizer(max_features=5000)
        
        self.topic_encoder = nn.Sequential(
            nn.Linear(num_topics, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.context_encoder = nn.LSTM(
            input_size=128,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )
        
        self.crf_layer = CRF(num_types, batch_first=True)
        
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
        if not values or not any(values):
            return np.zeros(self.num_topics)
        
        try:
            text_values = [str(v) for v in values if v]
            
            if not hasattr(self, 'fitted'):
                corpus = text_values[:1000]
                term_matrix = self.vectorizer.fit_transform(corpus)
                self.lda.fit(term_matrix)
                self.fitted = True
            
            term_matrix = self.vectorizer.transform(text_values)
            topic_dist = self.lda.transform(term_matrix)
            
            return np.mean(topic_dist, axis=0)
            
        except Exception as e:
            logger.debug(f"Topic extraction failed: {e}")
            return np.zeros(self.num_topics)
    
    def extract_context_features(self, table_data: Dict[str, List]) -> np.ndarray:
        context_features = []
        
        for column_name, values in table_data.items():
            column_topic = self.extract_topic_features(values[:100])
            context_features.append(column_topic)
        
        if context_features:
            return np.mean(context_features, axis=0)
        else:
            return np.zeros(self.num_topics)

class CRF(nn.Module):
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