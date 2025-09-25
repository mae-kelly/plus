import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class PatternRecognitionModel:
    
    def __init__(self, device: str = 'mps'):
        self.device = device
        self.transformer = HostnameTransformer(device=device)
        
        self.pattern_database = {
            'learned_patterns': [],
            'pattern_embeddings': None,
            'pattern_frequencies': {}
        }
        
        self.training_config = {
            'epochs': 100,
            'batch_size': 256,
            'learning_rate': 1e-4,
            'warmup_steps': 1000
        }
        
    async def learn_patterns(self, samples: List[str]):
        if not samples:
            return
            
        logger.info(f"Learning patterns from {len(samples)} samples")
        
        for sample in samples:
            if sample and sample not in self.pattern_database['learned_patterns']:
                self.pattern_database['learned_patterns'].append(sample)
                
                pattern_key = self._get_pattern_key(sample)
                if pattern_key not in self.pattern_database['pattern_frequencies']:
                    self.pattern_database['pattern_frequencies'][pattern_key] = 0
                self.pattern_database['pattern_frequencies'][pattern_key] += 1
        
        await self._train_transformer(samples)
        
        logger.info(f"Learned {len(self.pattern_database['learned_patterns'])} unique patterns")
        
    def _get_pattern_key(self, hostname: str) -> str:
        pattern = []
        
        dot_count = hostname.count('.')
        pattern.append(f"dots:{dot_count}")
        
        dash_count = hostname.count('-')
        pattern.append(f"dashes:{dash_count}")
        
        length = len(hostname)
        if length < 10:
            pattern.append("short")
        elif length < 30:
            pattern.append("medium")
        else:
            pattern.append("long")
            
        has_digits = any(c.isdigit() for c in hostname)
        has_upper = any(c.isupper() for c in hostname)
        
        if has_digits:
            pattern.append("has_digits")
        if has_upper:
            pattern.append("has_upper")
            
        return "_".join(pattern)
        
    async def _train_transformer(self, samples: List[str]):
        train_data = self._prepare_training_data(samples)
        
        if not train_data:
            return
            
        optimizer = torch.optim.AdamW(
            self.transformer.parameters(),
            lr=self.training_config['learning_rate']
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2
        )
        
        self.transformer.train()
        
        for epoch in range(self.training_config['epochs']):
            epoch_loss = 0
            
            for batch in self._get_batches(train_data, self.training_config['batch_size']):
                optimizer.zero_grad()
                
                outputs = self.transformer(batch['input'])
                loss = self._compute_loss(outputs, batch['target'])
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.transformer.parameters(), 1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                
            scheduler.step()
            
            if epoch % 10 == 0:
                logger.debug(f"Epoch {epoch}, Loss: {epoch_loss:.4f}")
                
    def _prepare_training_data(self, samples: List[str]) -> List[Dict]:
        data = []
        for sample in samples:
            if sample:
                tensor_repr = self._text_to_tensor(sample)
                data.append({
                    'input': tensor_repr,
                    'target': 1
                })
                
        return data
        
    def _text_to_tensor(self, text: str) -> torch.Tensor:
        max_len = 128
        vocab_size = 256
        
        indices = [ord(c) if ord(c) < vocab_size else 0 for c in text[:max_len]]
        
        if len(indices) < max_len:
            indices += [0] * (max_len - len(indices))
            
        return torch.tensor(indices, device=self.device)
        
    def _get_batches(self, data: List[Dict], batch_size: int):
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            
            inputs = torch.stack([item['input'] for item in batch])
            targets = torch.tensor([item['target'] for item in batch], device=self.device)
            
            yield {
                'input': inputs,
                'target': targets
            }
            
    def _compute_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        criterion = nn.BCELoss()
        return criterion(outputs.squeeze(), targets.float())


class HostnameTransformer(nn.Module):
    
    def __init__(self, device: str = 'mps'):
        super().__init__()
        
        self.device = device
        
        self.d_model = 512
        self.nhead = 8
        self.num_layers = 6
        self.vocab_size = 256
        self.max_len = 128
        
        self.embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.pos_encoding = PositionalEncoding(self.d_model, self.max_len)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=2048,
            dropout=0.1,
            activation='relu',
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_layers
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(self.d_model, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        self.to(device)
        
    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        x = self.transformer_encoder(x)
        x = torch.mean(x, dim=1)
        output = self.classifier(x)
        return output


class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x