import torch
import numpy as np
from typing import Dict, List, Any
import asyncio

from models.sherlock_model import SherlockModel
from models.sato_model import SatoModel
from models.doduo_model import DoduoModel
from models.llm_classifier import LLMClassifier

class EnsemblePredictor:
    def __init__(self, config: Dict, device: str):
        self.config = config
        self.device = device
        
        self.sherlock = SherlockModel(
            input_dim=config['model_config']['sherlock_features'],
            device=device
        )
        
        self.sato = SatoModel(
            num_topics=config['model_config']['sato_topics'],
            device=device
        )
        
        self.doduo = DoduoModel(device=device)
        
        self.llm_classifier = LLMClassifier(config)
        
        self.ensemble_weights = {
            'sherlock': 0.25,
            'sato': 0.30,
            'doduo': 0.35,
            'llm': 0.10
        }
        
        self.type_mapping = self._create_type_mapping()
    
    async def classify_columns(self, column_metadata: Dict, host_data: Dict) -> Dict[str, Dict]:
        classifications = {}
        
        for column_name, metadata in column_metadata.items():
            predictions = await self._get_ensemble_predictions(
                column_name,
                metadata['samples'],
                metadata
            )
            
            classifications[column_name] = self._aggregate_predictions(predictions)
        
        return classifications
    
    async def predict_single(self, column_name: str, value: Any, features: np.ndarray) -> List[Dict]:
        predictions = []
        
        with torch.no_grad():
            sherlock_features = torch.tensor(features, device=self.device).unsqueeze(0)
            sherlock_logits, sherlock_conf = self.sherlock(sherlock_features)
            
            sherlock_probs = torch.softmax(sherlock_logits, dim=-1)
            top_k_probs, top_k_indices = torch.topk(sherlock_probs, k=3)
            
            for prob, idx in zip(top_k_probs[0], top_k_indices[0]):
                predictions.append({
                    'type': self._idx_to_type(idx.item()),
                    'confidence': prob.item() * sherlock_conf.item(),
                    'model': 'sherlock'
                })
        
        return predictions
    
    async def _get_ensemble_predictions(self, column_name: str, values: List, metadata: Dict) -> List[Dict]:
        predictions = []
        
        sherlock_pred = await self._get_sherlock_prediction(column_name, values)
        predictions.append(sherlock_pred)
        
        sato_pred = await self._get_sato_prediction(values, metadata)
        predictions.append(sato_pred)
        
        doduo_pred = await self._get_doduo_prediction(column_name, values)
        predictions.append(doduo_pred)
        
        if self.config.get('use_llm', False):
            llm_pred = await self.llm_classifier.classify_with_llm(column_name, values[:20])
            predictions.append({
                'model': 'llm',
                'type': llm_pred['type'],
                'confidence': llm_pred['confidence']
            })
        
        return predictions
    
    async def _get_sherlock_prediction(self, column_name: str, values: List) -> Dict:
        features = self.sherlock.extract_features(column_name, values)
        features_tensor = torch.tensor(features, device=self.device).unsqueeze(0)
        
        with torch.no_grad():
            logits, confidence = self.sherlock(features_tensor)
            probs = torch.softmax(logits, dim=-1)
            type_idx = torch.argmax(probs).item()
        
        return {
            'model': 'sherlock',
            'type': self._idx_to_type(type_idx),
            'confidence': probs[0, type_idx].item() * confidence.item()
        }
    
    async def _get_sato_prediction(self, values: List, metadata: Dict) -> Dict:
        topic_features = self.sato.extract_topic_features([str(v) for v in values if v])
        topic_tensor = torch.tensor(topic_features, device=self.device, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            logits = self.sato(topic_tensor)
            probs = torch.softmax(logits, dim=-1)
            type_idx = torch.argmax(probs).item()
        
        return {
            'model': 'sato',
            'type': self._idx_to_type(type_idx),
            'confidence': probs[0, type_idx].item()
        }
    
    async def _get_doduo_prediction(self, column_name: str, values: List) -> Dict:
        table_data = {column_name: values}
        column_types, _ = self.doduo.process_table(table_data)
        
        with torch.no_grad():
            probs = torch.softmax(column_types, dim=-1)
            type_idx = torch.argmax(probs).item()
        
        return {
            'model': 'doduo',
            'type': self._idx_to_type(type_idx),
            'confidence': probs[0, type_idx].item()
        }
    
    def _aggregate_predictions(self, predictions: List[Dict]) -> Dict:
        type_scores = {}
        
        for pred in predictions:
            model = pred['model']
            pred_type = pred['type']
            confidence = pred['confidence']
            
            weight = self.ensemble_weights.get(model, 0.1)
            score = confidence * weight
            
            if pred_type not in type_scores:
                type_scores[pred_type] = 0
            type_scores[pred_type] += score
        
        if not type_scores:
            return {'type': 'unknown', 'confidence': 0.0}
        
        best_type = max(type_scores, key=type_scores.get)
        total_weight = sum(self.ensemble_weights.values())
        
        return {
            'type': best_type,
            'confidence': type_scores[best_type] / total_weight,
            'scores': type_scores,
            'relationships': self._infer_relationships(best_type)
        }
    
    def _create_type_mapping(self) -> Dict[int, str]:
        types = [
            'unknown', 'hostname', 'ip_address', 'email', 'phone', 'address',
            'person_name', 'organization', 'timestamp', 'date', 'time',
            'numeric_id', 'text_id', 'amount', 'percentage', 'url',
            'file_path', 'country', 'city', 'state', 'zip_code',
            'latitude', 'longitude', 'currency', 'language', 'boolean',
            'json', 'xml', 'base64', 'md5', 'sha1', 'sha256',
            'credit_card', 'ssn', 'passport', 'driver_license', 'vin',
            'isbn', 'ean', 'upc', 'color', 'size', 'weight',
            'temperature', 'pressure', 'speed', 'version', 'user_agent',
            'referrer', 'cookie', 'session_id', 'transaction_id', 'order_id',
            'product_id', 'customer_id', 'employee_id', 'department', 'title',
            'description', 'comment', 'note', 'tag', 'category',
            'status', 'priority', 'severity', 'environment', 'region',
            'cluster', 'datacenter', 'rack', 'server_type', 'os',
            'application', 'service', 'api_key', 'token', 'password_hash'
        ]
        
        return {i: t for i, t in enumerate(types[:78])}
    
    def _idx_to_type(self, idx: int) -> str:
        return self.type_mapping.get(idx, 'unknown')
    
    def _infer_relationships(self, column_type: str) -> List[str]:
        relationships = {
            'hostname': ['ip_address', 'datacenter', 'rack', 'os', 'environment'],
            'ip_address': ['hostname', 'datacenter', 'subnet', 'vlan'],
            'customer_id': ['order_id', 'transaction_id', 'email', 'phone'],
            'employee_id': ['department', 'title', 'email', 'phone'],
            'timestamp': ['user_id', 'session_id', 'event_type', 'status']
        }
        
        return relationships.get(column_type, [])