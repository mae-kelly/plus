import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from collections import Counter
import logging
import re

logger = logging.getLogger(__name__)

class PatternRecognitionModel:
    """
    Learns what hostnames look like by training on known examples.
    
    Uses a transformer model because hostnames have sequential patterns
    (like subdomain.domain.tld) that transformers are good at catching.
    """
    
    def __init__(self, device: str = 'mps'):
        self.device = device
        
        # Don't create the model until we know what we're dealing with
        self.transformer = None
        self.is_trained = False
        
        # Keep track of what we've seen
        self.pattern_database = {
            'learned_patterns': [],
            'pattern_embeddings': None,
            'pattern_frequencies': {},
            'common_formats': Counter(),  # Track FQDN vs short vs IP etc
            'length_stats': {'min': 999, 'max': 0, 'avg': 0}
        }
        
        # Training params - lower epochs for faster training
        self.training_config = {
            'epochs': 20,  # reduced from 100 - hostnames don't need that much
            'batch_size': 64,  # smaller batches work fine
            'learning_rate': 5e-4,  # bit higher for faster convergence
            'min_samples': 10  # need at least this many to bother training
        }
        
        # Cache predictions to avoid recalculating
        self.prediction_cache = {}
        
    async def learn_patterns(self, samples: List[str]):
        """
        Figure out what hostnames look like from examples.
        Skip training if we don't have enough data.
        """
        if not samples:
            logger.warning("No samples provided for pattern learning")
            return
            
        # Clean up the samples first
        cleaned_samples = []
        for sample in samples:
            if not sample:
                continue
            sample = str(sample).strip()
            if len(sample) < 2 or len(sample) > 253:  # DNS limits
                continue
            cleaned_samples.append(sample)
        
        if len(cleaned_samples) < self.training_config['min_samples']:
            logger.info(f"Only {len(cleaned_samples)} samples, using heuristics only")
            self._analyze_patterns_only(cleaned_samples)
            return
            
        logger.info(f"Learning from {len(cleaned_samples)} hostname samples")
        
        # Analyze patterns first (fast)
        self._analyze_patterns_only(cleaned_samples)
        
        # Only train neural net if we have enough variety
        unique_patterns = len(self.pattern_database['pattern_frequencies'])
        if unique_patterns < 3:
            logger.info("Samples too uniform, skipping neural training")
            return
            
        # Now train the transformer
        await self._train_transformer(cleaned_samples)
        self.is_trained = True
        
        logger.info(f"Training complete. Learned {unique_patterns} pattern types")
        
    def _analyze_patterns_only(self, samples: List[str]):
        """
        Quick pattern analysis without neural net.
        This is fast and catches obvious patterns.
        """
        lengths = []
        
        for sample in samples:
            # Track lengths
            length = len(sample)
            lengths.append(length)
            if length < self.pattern_database['length_stats']['min']:
                self.pattern_database['length_stats']['min'] = length
            if length > self.pattern_database['length_stats']['max']:
                self.pattern_database['length_stats']['max'] = length
            
            # Classify format type
            if '.' in sample and sample.count('.') >= 2:
                self.pattern_database['common_formats']['fqdn'] += 1
            elif re.match(r'^\d+\.\d+\.\d+\.\d+$', sample):
                self.pattern_database['common_formats']['ipv4'] += 1
            elif ':' in sample and sample.count(':') > 1:
                self.pattern_database['common_formats']['ipv6'] += 1
            elif '-' in sample or '_' in sample:
                self.pattern_database['common_formats']['hyphenated'] += 1
            else:
                self.pattern_database['common_formats']['simple'] += 1
            
            # Store pattern signature
            pattern_key = self._get_pattern_key(sample)
            if pattern_key not in self.pattern_database['pattern_frequencies']:
                self.pattern_database['pattern_frequencies'][pattern_key] = 0
            self.pattern_database['pattern_frequencies'][pattern_key] += 1
            
            # Keep some examples
            if len(self.pattern_database['learned_patterns']) < 1000:
                self.pattern_database['learned_patterns'].append(sample.lower())
        
        if lengths:
            self.pattern_database['length_stats']['avg'] = sum(lengths) / len(lengths)
        
    def _get_pattern_key(self, hostname: str) -> str:
        """
        Create a signature for this hostname's structure.
        Used to group similar patterns.
        """
        parts = []
        
        # Check basic structure
        if '.' in hostname:
            parts.append(f"dots{hostname.count('.')}")
        if '-' in hostname:
            parts.append(f"dash{min(hostname.count('-'), 3)}")  # cap at 3
        if '_' in hostname:
            parts.append("underscore")
            
        # Length bucket
        length = len(hostname)
        if length < 8:
            parts.append("tiny")
        elif length < 16:
            parts.append("short")
        elif length < 32:
            parts.append("medium")
        elif length < 64:
            parts.append("long")
        else:
            parts.append("huge")
            
        # Character types
        if any(c.isdigit() for c in hostname):
            parts.append("numeric")
        if hostname != hostname.lower():
            parts.append("mixed_case")
        if re.search(r'[^a-zA-Z0-9.\-_]', hostname):
            parts.append("special_chars")
            
        return "_".join(parts) if parts else "simple"
        
    async def _train_transformer(self, samples: List[str]):
        """
        Train the neural net. Only called if we have good training data.
        """
        
        # Initialize model now that we're actually training
        self.transformer = HostnameTransformer(device=self.device)
        
        # Prepare data - need positive and negative examples
        train_data = self._prepare_training_data(samples)
        
        if len(train_data) < 20:
            logger.warning("Not enough training data generated")
            return
            
        optimizer = torch.optim.AdamW(
            self.transformer.parameters(),
            lr=self.training_config['learning_rate'],
            weight_decay=0.01
        )
        
        self.transformer.train()
        
        # Quick training loop - hostnames aren't that complex
        for epoch in range(self.training_config['epochs']):
            epoch_loss = 0
            batch_count = 0
            
            for batch in self._get_batches(train_data, self.training_config['batch_size']):
                optimizer.zero_grad()
                
                outputs = self.transformer(batch['input'])
                loss = F.binary_cross_entropy(outputs.squeeze(), batch['target'].float())
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.transformer.parameters(), 1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                batch_count += 1
                
            if epoch % 5 == 0 and batch_count > 0:
                avg_loss = epoch_loss / batch_count
                logger.debug(f"Epoch {epoch}, Avg Loss: {avg_loss:.4f}")
                
                # Early stopping if loss is good enough
                if avg_loss < 0.1:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
                    
        self.transformer.eval()
        
    def _prepare_training_data(self, samples: List[str]) -> List[Dict]:
        """
        Create training data with positive (real) and negative (fake) examples.
        This teaches the model what's NOT a hostname too.
        """
        data = []
        
        # Positive examples (real hostnames)
        for sample in samples[:500]:  # cap it
            if sample:
                tensor_repr = self._text_to_tensor(sample)
                data.append({
                    'input': tensor_repr,
                    'target': 1  # real hostname
                })
                
        # Generate negative examples (definitely not hostnames)
        negative_examples = [
            "this is a sentence not a hostname",
            "random text here",
            "!!!###$$$%%%",
            "1234567890" * 10,  # too long
            "a",  # too short
            "../../../etc/passwd",  # path traversal
            "hello world.txt",
            "user@email.com",  # email not hostname
            "http://website.com",  # URL not hostname
            "2024-01-15",  # date
            "(555) 123-4567",  # phone
        ]
        
        # Add some random garbage
        import random
        import string
        for _ in range(min(len(samples), 100)):
            # Random strings that don't look like hostnames
            garbage = ''.join(random.choices(string.ascii_letters + string.digits + " !@#$%", 
                                           k=random.randint(20, 50)))
            negative_examples.append(garbage)
            
        for neg_sample in negative_examples:
            tensor_repr = self._text_to_tensor(neg_sample)
            data.append({
                'input': tensor_repr,
                'target': 0  # not a hostname
            })
            
        # Shuffle so training sees mixed examples
        random.shuffle(data)
        return data
        
    def _text_to_tensor(self, text: str) -> torch.Tensor:
        """Convert text to tensor of character indices."""
        max_len = 128
        
        # Convert to indices (ASCII values)
        indices = []
        for c in text[:max_len]:
            idx = ord(c) if ord(c) < 256 else 0  # unknown char -> 0
            indices.append(idx)
            
        # Pad if needed
        if len(indices) < max_len:
            indices.extend([0] * (max_len - len(indices)))
            
        return torch.tensor(indices, device=self.device, dtype=torch.long)
        
    def _get_batches(self, data: List[Dict], batch_size: int):
        """Yield batches for training."""
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            
            if not batch:
                continue
                
            inputs = torch.stack([item['input'] for item in batch])
            targets = torch.tensor([item['target'] for item in batch], 
                                  device=self.device, dtype=torch.float)
            
            yield {
                'input': inputs,
                'target': targets
            }
            
    def predict(self, value: str) -> float:
        """
        Score how likely this is to be a hostname (0-1).
        Uses neural net if trained, otherwise heuristics.
        """
        if not value or len(value) < 2:
            return 0.0
            
        # Check cache first
        if value in self.prediction_cache:
            return self.prediction_cache[value]
            
        score = 0.0
        
        # If we have a trained model, use it
        if self.is_trained and self.transformer:
            try:
                self.transformer.eval()
                with torch.no_grad():
                    tensor_input = self._text_to_tensor(value).unsqueeze(0)
                    output = self.transformer(tensor_input)
                    score = output.item()
            except Exception as e:
                logger.debug(f"Neural prediction failed: {e}, using heuristics")
                score = self._heuristic_score(value)
        else:
            # Fall back to heuristics
            score = self._heuristic_score(value)
            
        # Cache the result
        self.prediction_cache[value] = score
        return score
        
    def _heuristic_score(self, value: str) -> float:
        """
        Quick scoring based on patterns we've seen.
        This runs when we have no trained model.
        """
        score = 0.5  # start neutral
        
        # Check length
        length = len(value)
        if self.pattern_database['length_stats']['avg'] > 0:
            avg_len = self.pattern_database['length_stats']['avg']
            if abs(length - avg_len) < 10:
                score += 0.1
        
        # Check if it matches common formats we've seen
        if self.pattern_database['common_formats']:
            most_common = self.pattern_database['common_formats'].most_common(1)[0][0]
            
            if most_common == 'fqdn' and '.' in value:
                score += 0.2
            elif most_common == 'ipv4' and re.match(r'^\d+\.\d+\.\d+\.\d+$', value):
                score = 0.95  # IPs are pretty obvious
            elif most_common == 'hyphenated' and '-' in value:
                score += 0.15
                
        # Check pattern match
        pattern_key = self._get_pattern_key(value)
        if pattern_key in self.pattern_database['pattern_frequencies']:
            score += 0.2
            
        # Penalize weird stuff
        if any(c in value for c in ['!', '@', '#', '$', '%', '^', '&', '*', '(', ')', ' ']):
            score -= 0.3
            
        return max(0.0, min(1.0, score))


class HostnameTransformer(nn.Module):
    """
    Transformer model for learning hostname patterns.
    Smaller than typical transformers because hostnames are short.
    """
    
    def __init__(self, device: str = 'mps'):
        super().__init__()
        
        self.device = device
        
        # Small model - hostnames aren't Shakespeare
        self.d_model = 64  # smaller embeddings
        self.nhead = 4  # fewer attention heads
        self.num_layers = 2  # shallow network
        self.vocab_size = 256  # ASCII
        self.max_len = 128  # max hostname length
        
        # Layers
        self.embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.pos_encoding = PositionalEncoding(self.d_model, self.max_len)
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=256,
            dropout=0.1,
            activation='gelu',  # gelu often better than relu
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_layers
        )
        
        # Output head
        self.classifier = nn.Sequential(
            nn.Linear(self.d_model, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        self.to(device)
        
    def forward(self, x):
        # x is (batch, seq_len) of character indices
        x = self.embedding(x)  # -> (batch, seq_len, d_model)
        x = self.pos_encoding(x)
        
        # Create attention mask to ignore padding (0s)
        mask = (x == 0).all(dim=-1)  # find padded positions
        
        x = self.transformer_encoder(x, src_key_padding_mask=mask)
        
        # Global average pooling
        x = x.masked_fill(mask.unsqueeze(-1), 0)  # zero out padding
        x = x.sum(dim=1) / (~mask).sum(dim=1, keepdim=True).clamp(min=1)
        
        return self.classifier(x)


class PositionalEncoding(nn.Module):
    """
    Standard positional encoding for transformers.
    Helps the model understand position in the sequence.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Create sinusoidal patterns
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model > 1:
            pe[:, 1::2] = torch.cos(position * div_term[:d_model//2])  # handle odd d_model
        
        pe = pe.unsqueeze(0)  # add batch dimension
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # Add positional encoding to embeddings
        return x + self.pe[:, :x.size(1), :]