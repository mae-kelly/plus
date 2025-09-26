import json
from typing import List, Dict, Any
from snorkel.labeling import LabelingFunction, PandasLFApplier
from snorkel.labeling.model import LabelModel
import numpy as np
import pandas as pd
import logging
import re

logger = logging.getLogger(__name__)

class RuleBasedClassifier:
    """Rule-based classifier to replace LLM"""
    def __init__(self):
        self.patterns = self._initialize_patterns()
        
    def _initialize_patterns(self):
        """Initialize classification patterns"""
        return {
            'hostname': [
                (r'^[a-zA-Z0-9][a-zA-Z0-9-]{0,61}[a-zA-Z0-9]?$', 0.9),
                (r'server|host|node|machine', 0.7),
            ],
            'ip_address': [
                (r'^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$', 0.95),
                (r'^(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$', 0.95),
            ],
            'email': [
                (r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', 0.95),
            ],
            'timestamp': [
                (r'^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}', 0.9),
                (r'^\d{10,13}$', 0.7),
            ],
            'numeric_id': [
                (r'^\d+$', 0.8),
            ],
            'url': [
                (r'^https?://[^\s]+$', 0.95),
            ],
            'phone': [
                (r'^(\+\d{1,3})?[\s.-]?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}$', 0.9),
            ]
        }
    
    def classify(self, column_name: str, values: List[Any]) -> Dict[str, Any]:
        """Classify column based on patterns"""
        scores = {}
        
        # Clean values
        str_values = [str(v).strip() for v in values[:100] if v is not None]
        
        if not str_values:
            return {'type': 'unknown', 'confidence': 0.0}
        
        # Check each pattern
        for data_type, patterns in self.patterns.items():
            type_score = 0
            for pattern, weight in patterns:
                matches = sum(1 for v in str_values if re.match(pattern, v, re.I))
                match_ratio = matches / len(str_values)
                type_score = max(type_score, match_ratio * weight)
            
            if type_score > 0:
                scores[data_type] = type_score
        
        # Check column name hints
        column_lower = column_name.lower()
        if 'host' in column_lower or 'server' in column_lower:
            scores['hostname'] = max(scores.get('hostname', 0), 0.7)
        elif 'ip' in column_lower:
            scores['ip_address'] = max(scores.get('ip_address', 0), 0.7)
        elif 'email' in column_lower or 'mail' in column_lower:
            scores['email'] = max(scores.get('email', 0), 0.7)
        elif 'time' in column_lower or 'date' in column_lower:
            scores['timestamp'] = max(scores.get('timestamp', 0), 0.6)
        elif 'id' in column_lower or 'key' in column_lower:
            scores['numeric_id'] = max(scores.get('numeric_id', 0), 0.5)
        
        # Get best match
        if scores:
            best_type = max(scores, key=scores.get)
            return {
                'type': best_type,
                'confidence': scores[best_type],
                'reasoning': f"Pattern matching confidence: {scores[best_type]:.2f}"
            }
        
        return {'type': 'unknown', 'confidence': 0.0}

class LLMClassifier:
    def __init__(self, config: Dict):
        # Remove OpenAI API key usage
        self.model = config['model_config'].get('llm_model', 'rule-based')
        self.batch_size = config['model_config'].get('llm_batch_size', 50)
        self.weak_supervision_ratio = config['model_config'].get('weak_supervision_ratio', 0.013)
        
        # Use rule-based classifier instead of OpenAI
        self.rule_classifier = RuleBasedClassifier()
        
        self.label_model = LabelModel(cardinality=78, verbose=False)
        self.labeling_functions = []
    
    async def generate_labeling_functions(self, column_samples: Dict[str, List]) -> List[LabelingFunction]:
        """Generate labeling functions based on patterns"""
        functions = []
        
        for column_name, values in list(column_samples.items())[:10]:
            # Create pattern-based labeling function
            def create_lf(col_name, sample_values):
                def labeling_function(series):
                    # Simple pattern matching
                    str_values = series.astype(str)
                    
                    # Check for IPs
                    if str_values.str.match(r'^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$').mean() > 0.8:
                        return 1  # IP address
                    
                    # Check for emails
                    if str_values.str.contains('@').mean() > 0.8:
                        return 2  # Email
                    
                    # Check for timestamps
                    if str_values.str.match(r'^\d{4}-\d{2}-\d{2}').mean() > 0.8:
                        return 3  # Timestamp
                    
                    # Check for numeric IDs
                    if str_values.str.match(r'^\d+$').mean() > 0.8:
                        return 4  # Numeric ID
                    
                    # Check column name
                    if 'host' in col_name.lower():
                        return 0  # Hostname
                    
                    return -1  # Abstain
                
                return labeling_function
            
            lf = create_lf(column_name, values)
            lf_wrapper = LabelingFunction(name=f"lf_{len(functions)}", f=lf)
            functions.append(lf_wrapper)
        
        self.labeling_functions.extend(functions)
        return functions
    
    async def classify_with_llm(self, column_name: str, values: List[Any]) -> Dict[str, Any]:
        """Classify using rule-based system instead of LLM"""
        try:
            result = self.rule_classifier.classify(column_name, values)
            return result
        except Exception as e:
            logger.error(f"Classification failed: {e}")
            return {'type': 'unknown', 'confidence': 0.0}
    
    def train_weak_supervision(self, df: pd.DataFrame):
        """Train weak supervision model"""
        if not self.labeling_functions:
            return None, None
        
        applier = PandasLFApplier(self.labeling_functions)
        L = applier.apply(df)
        
        self.label_model.fit(L)
        
        predictions = self.label_model.predict(L)
        confidences = self.label_model.predict_proba(L)
        
        return predictions, confidences