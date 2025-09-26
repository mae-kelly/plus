import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Any
import re
from collections import Counter

class LocalTransformerEncoder(nn.Module):
    """Custom transformer encoder without external dependencies"""
    def __init__(self, vocab_size: int = 30000, hidden_size: int = 768, num_layers: int = 6):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(512, hidden_size)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=12,
            dim_feedforward=3072,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Layer norm
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # Pooler
        self.pooler = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh()
        )
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None):
        seq_length = input_ids.shape[1]
        position_ids = torch.arange(seq_length, device=input_ids.device).expand_as(input_ids)
        
        # Embeddings
        token_embeds = self.token_embedding(input_ids)
        position_embeds = self.position_embedding(position_ids)
        embeddings = token_embeds + position_embeds
        
        # Transformer
        if attention_mask is not None:
            attention_mask = attention_mask.float()
            attention_mask = (1.0 - attention_mask) * -10000.0
        
        hidden_states = self.transformer(embeddings, src_key_padding_mask=attention_mask)
        hidden_states = self.layer_norm(hidden_states)
        
        # Pool the first token (CLS token)
        pooled_output = self.pooler(hidden_states[:, 0])
        
        return hidden_states, pooled_output

class SimpleTokenizer:
    """Simple tokenizer without external dependencies"""
    def __init__(self, vocab_size: int = 30000):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.reverse_vocab = {}
        self._build_basic_vocab()
        
    def _build_basic_vocab(self):
        """Build a basic vocabulary"""
        # Special tokens
        special_tokens = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', '[COL]', '[VAL]']
        for i, token in enumerate(special_tokens):
            self.vocab[token] = i
            self.reverse_vocab[i] = token
        
        # Common words and characters
        current_id = len(special_tokens)
        
        # Add common English words (simplified)
        common_words = ['the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i',
                       'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
                       'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she',
                       'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their',
                       'what', 'so', 'up', 'out', 'if', 'about', 'who', 'get', 'which', 'go',
                       'me', 'when', 'make', 'can', 'like', 'time', 'no', 'just', 'him', 'know']
        
        for word in common_words:
            if current_id < self.vocab_size:
                self.vocab[word] = current_id
                self.reverse_vocab[current_id] = word
                current_id += 1
        
        # Add single characters
        for i in range(256):
            char = chr(i)
            if char not in self.vocab and current_id < self.vocab_size:
                self.vocab[char] = current_id
                self.reverse_vocab[current_id] = char
                current_id += 1
    
    def tokenize(self, text: str, max_length: int = 512) -> Dict[str, torch.Tensor]:
        """Tokenize text into IDs"""
        # Simple word-level tokenization
        tokens = ['[CLS]']
        
        # Split by whitespace and punctuation
        words = re.findall(r'\w+|[^\w\s]', text.lower())
        
        for word in words[:max_length-2]:  # Reserve space for [CLS] and [SEP]
            if word in self.vocab:
                tokens.append(word)
            else:
                # Character-level fallback
                for char in word:
                    if char in self.vocab:
                        tokens.append(char)
                    else:
                        tokens.append('[UNK]')
        
        tokens.append('[SEP]')
        
        # Convert to IDs
        input_ids = [self.vocab.get(token, self.vocab['[UNK]']) for token in tokens]
        
        # Pad to max_length
        attention_mask = [1] * len(input_ids)
        
        while len(input_ids) < max_length:
            input_ids.append(self.vocab['[PAD]'])
            attention_mask.append(0)
        
        input_ids = input_ids[:max_length]
        attention_mask = attention_mask[:max_length]
        
        return {
            'input_ids': torch.tensor([input_ids]),
            'attention_mask': torch.tensor([attention_mask])
        }

class DoduoModel(nn.Module):
    def __init__(self, num_types: int = 78, device: str = 'mps'):
        super().__init__()
        self.device = device
        
        # Custom tokenizer and encoder
        self.tokenizer = SimpleTokenizer()
        self.encoder = LocalTransformerEncoder(vocab_size=30000, hidden_size=768)
        
        hidden_size = 768
        
        # Column type classification head
        self.column_type_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, num_types)
        )
        
        # Column relation head
        self.column_relation_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 4)
        )
        
        # Additional feature extractors
        self.statistical_encoder = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128)
        )
        
        # Combined classifier
        self.final_classifier = nn.Sequential(
            nn.Linear(hidden_size + 128, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_types)
        )
        
        self.to(device)
    
    def forward(self, table_tokens, statistical_features=None):
        # Get transformer output
        hidden_states, pooled_output = self.encoder(
            table_tokens['input_ids'],
            table_tokens.get('attention_mask')
        )
        
        # Get column type predictions
        column_types = self.column_type_head(pooled_output)
        
        # If statistical features provided, combine them
        if statistical_features is not None:
            stat_encoded = self.statistical_encoder(statistical_features)
            combined = torch.cat([pooled_output, stat_encoded], dim=-1)
            column_types = self.final_classifier(combined)
        
        return column_types, None
    
    def process_table(self, table_data: Dict[str, List]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process table data and return predictions"""
        # Serialize table
        serialized_table = self._serialize_table(table_data)
        
        # Tokenize
        tokens = self.tokenizer.tokenize(serialized_table, max_length=512)
        tokens = {k: v.to(self.device) for k, v in tokens.items()}
        
        # Extract statistical features
        stat_features = self._extract_statistical_features(table_data)
        stat_features = torch.tensor(stat_features, device=self.device).unsqueeze(0)
        
        with torch.no_grad():
            column_types, column_relations = self.forward(tokens, stat_features)
        
        return column_types, column_relations
    
    def _serialize_table(self, table_data: Dict[str, List]) -> str:
        """Serialize table data for processing"""
        serialized_parts = []
        
        for column_name, values in list(table_data.items())[:50]:
            # Sample values
            sample_values = values[:8]
            value_str = ' | '.join([str(v) for v in sample_values if v])
            serialized_parts.append(f"[COL] {column_name} [VAL] {value_str}")
        
        return ' '.join(serialized_parts)[:2048]
    
    def _extract_statistical_features(self, table_data: Dict[str, List]) -> List[float]:
        """Extract statistical features from table"""
        features = []
        
        for column_name, values in list(table_data.items())[:10]:
            # Basic statistics
            str_values = [str(v) if v is not None else '' for v in values[:100]]
            
            if str_values:
                lengths = [len(v) for v in str_values]
                features.extend([
                    np.mean(lengths) if lengths else 0,
                    np.std(lengths) if len(lengths) > 1 else 0,
                    len(set(str_values)) / len(str_values),  # Uniqueness
                    sum(1 for v in str_values if v.isdigit()) / len(str_values),  # Numeric ratio
                    sum(1 for v in str_values if '@' in v) / len(str_values),  # Email-like ratio
                ])
            else:
                features.extend([0, 0, 0, 0, 0])
        
        # Pad to fixed size
        while len(features) < 100:
            features.append(0)
        
        return features[:100]
    
    def extract_column_embeddings(self, table_data: Dict[str, List]) -> Dict[str, np.ndarray]:
        """Extract embeddings for each column"""
        embeddings = {}
        
        for column_name, values in table_data.items():
            # Create column text
            column_text = f"{column_name} " + ' '.join([str(v) for v in values[:10] if v])
            
            # Tokenize
            tokens = self.tokenizer.tokenize(column_text, max_length=128)
            tokens = {k: v.to(self.device) for k, v in tokens.items()}
            
            with torch.no_grad():
                hidden_states, pooled_output = self.encoder(
                    tokens['input_ids'],
                    tokens.get('attention_mask')
                )
                embedding = pooled_output.cpu().numpy()
            
            embeddings[column_name] = embedding
        
        return embeddings
    
    def predict_column_relations(self, col1_embedding: np.ndarray, col2_embedding: np.ndarray) -> str:
        """Predict relationship between two columns"""
        combined = torch.tensor(
            np.concatenate([col1_embedding, col2_embedding]),
            device=self.device
        ).unsqueeze(0)
        
        with torch.no_grad():
            relation_logits = self.column_relation_head(combined)
            relation_type = torch.argmax(relation_logits).item()
        
        relation_types = ['none', 'foreign_key', 'same_entity', 'hierarchical']
        return relation_types[relation_type]