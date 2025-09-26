import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import numpy as np
from typing import List, Dict, Tuple, Any

class DoduoModel(nn.Module):
    def __init__(self, model_name: str = 'bert-base-uncased', num_types: int = 78, device: str = 'mps'):
        super().__init__()
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        
        hidden_size = self.encoder.config.hidden_size
        
        self.column_type_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, num_types)
        )
        
        self.column_relation_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 4)
        )
        
        self.to(device)
    
    def forward(self, table_tokens):
        outputs = self.encoder(**table_tokens)
        pooled_output = outputs.pooler_output
        
        column_types = self.column_type_head(pooled_output)
        
        return column_types, None
    
    def process_table(self, table_data: Dict[str, List]) -> Tuple[torch.Tensor, torch.Tensor]:
        serialized_table = self._serialize_table(table_data)
        
        tokens = self.tokenizer(
            serialized_table,
            truncation=True,
            max_length=512,
            padding='max_length',
            return_tensors='pt'
        )
        
        tokens = {k: v.to(self.device) for k, v in tokens.items()}
        
        with torch.no_grad():
            column_types, column_relations = self.forward(tokens)
        
        return column_types, column_relations
    
    def _serialize_table(self, table_data: Dict[str, List]) -> str:
        serialized_parts = []
        
        for column_name, values in list(table_data.items())[:50]:
            sample_values = values[:8]
            value_str = ' | '.join([str(v) for v in sample_values if v])
            serialized_parts.append(f"[COL] {column_name} [VAL] {value_str}")
        
        return ' '.join(serialized_parts)[:2048]
    
    def extract_column_embeddings(self, table_data: Dict[str, List]) -> Dict[str, np.ndarray]:
        embeddings = {}
        
        for column_name, values in table_data.items():
            column_text = f"{column_name} " + ' '.join([str(v) for v in values[:10] if v])
            
            tokens = self.tokenizer(
                column_text,
                truncation=True,
                max_length=128,
                padding='max_length',
                return_tensors='pt'
            )
            
            tokens = {k: v.to(self.device) for k, v in tokens.items()}
            
            with torch.no_grad():
                outputs = self.encoder(**tokens)
                embedding = outputs.pooler_output.cpu().numpy()
            
            embeddings[column_name] = embedding
        
        return embeddings
    
    def predict_column_relations(self, col1_embedding: np.ndarray, col2_embedding: np.ndarray) -> str:
        combined = torch.tensor(
            np.concatenate([col1_embedding, col2_embedding]),
            device=self.device
        )
        
        with torch.no_grad():
            relation_logits = self.column_relation_head(combined)
            relation_type = torch.argmax(relation_logits).item()
        
        relation_types = ['none', 'foreign_key', 'same_entity', 'hierarchical']
        return relation_types[relation_type]