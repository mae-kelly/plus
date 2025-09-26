import json
from typing import List, Dict, Any
import numpy as np
import pandas as pd
import logging
import re
from collections import Counter

logger = logging.getLogger(__name__)

class RuleBasedClassifier:
    """Rule-based classifier without external LLM dependencies"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.rules = self._initialize_rules()
        self.pattern_cache = {}
        
    def _initialize_rules(self) -> Dict[str, List[Dict]]:
        """Initialize classification rules"""
        return {
            'hostname': [
                {'pattern': r'^[a-zA-Z0-9][a-zA-Z0-9-]{0,61}[a-zA-Z0-9]?$', 'weight': 0.8},
                {'pattern': r'^[a-z]+-[a-z]+-\d+$', 'weight': 0.7},
                {'pattern': r'(server|host|node|machine)', 'weight': 0.6, 'type': 'contains'},
                {'column_name': ['host', 'hostname', 'server', 'machine'], 'weight': 0.9}
            ],
            'ip_address': [
                {'pattern': r'^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$', 'weight': 0.95},
                {'pattern': r'^(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$', 'weight': 0.95},
                {'column_name': ['ip', 'ipaddr', 'ip_address', 'address'], 'weight': 0.8}
            ],
            'email': [
                {'pattern': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', 'weight': 0.95},
                {'column_name': ['email', 'mail', 'email_address'], 'weight': 0.9}
            ],
            'timestamp': [
                {'pattern': r'^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}', 'weight': 0.9},
                {'pattern': r'^\d{10,13}$', 'weight': 0.7},  # Unix timestamp
                {'column_name': ['timestamp', 'created', 'modified', 'updated', 'datetime'], 'weight': 0.85}
            ],
            'date': [
                {'pattern': r'^\d{4}-\d{2}-\d{2}$', 'weight': 0.9},
                {'pattern': r'^\d{1,2}/\d{1,2}/\d{4}$', 'weight': 0.85},
                {'column_name': ['date', 'day', 'birth_date', 'start_date', 'end_date'], 'weight': 0.8}
            ],
            'phone': [
                {'pattern': r'^(\+\d{1,3})?[\s.-]?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}$', 'weight': 0.9},
                {'column_name': ['phone', 'telephone', 'mobile', 'cell'], 'weight': 0.85}
            ],
            'url': [
                {'pattern': r'^https?://[^\s]+$', 'weight': 0.95},
                {'pattern': r'^www\.[^\s]+$', 'weight': 0.8},
                {'column_name': ['url', 'link', 'website', 'uri'], 'weight': 0.85}
            ],
            'uuid': [
                {'pattern': r'^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$', 'weight': 0.95},
                {'column_name': ['uuid', 'guid', 'id'], 'weight': 0.7}
            ],
            'numeric_id': [
                {'pattern': r'^\d+$', 'weight': 0.6},
                {'column_name': ['id', 'identifier', 'key', 'number'], 'weight': 0.7},
                {'check': 'all_numeric', 'weight': 0.8}
            ],
            'amount': [
                {'pattern': r'^\$?\d+\.?\d*$', 'weight': 0.8},
                {'column_name': ['amount', 'price', 'cost', 'value', 'total', 'balance'], 'weight': 0.9},
                {'check': 'numeric_with_decimals', 'weight': 0.7}
            ],
            'percentage': [
                {'pattern': r'^\d+\.?\d*%?$', 'weight': 0.8},
                {'column_name': ['percentage', 'percent', 'rate', 'ratio'], 'weight': 0.85},
                {'check': 'range_0_100', 'weight': 0.7}
            ],
            'boolean': [
                {'values': ['true', 'false', '0', '1', 'yes', 'no', 't', 'f', 'y', 'n'], 'weight': 0.9},
                {'column_name': ['flag', 'is_', 'has_', 'enabled', 'active'], 'weight': 0.8}
            ],
            'environment': [
                {'values': ['prod', 'production', 'staging', 'dev', 'development', 'test', 'qa'], 'weight': 0.9},
                {'column_name': ['env', 'environment', 'stage'], 'weight': 0.85}
            ]
        }
    
    async def classify_with_rules(self, column_name: str, values: List[Any]) -> Dict[str, Any]:
        """Classify column using rule-based system"""
        
        scores = {}
        
        # Clean values
        str_values = [str(v).strip().lower() if v is not None else '' for v in values[:100]]
        non_empty_values = [v for v in str_values if v]
        
        if not non_empty_values:
            return {'type': 'unknown', 'confidence': 0.0, 'reasoning': 'No values to analyze'}
        
        # Check each type's rules
        for data_type, rules in self.rules.items():
            type_score = 0.0
            matches = 0
            
            for rule in rules:
                if 'pattern' in rule:
                    pattern_matches = self._check_pattern(non_empty_values, rule['pattern'], 
                                                         rule.get('type', 'match'))
                    if pattern_matches > 0:
                        type_score += pattern_matches * rule['weight']
                        matches += 1
                
                elif 'column_name' in rule:
                    if self._check_column_name(column_name, rule['column_name']):
                        type_score += rule['weight']
                        matches += 1
                
                elif 'values' in rule:
                    value_matches = self._check_values(non_empty_values, rule['values'])
                    if value_matches > 0:
                        type_score += value_matches * rule['weight']
                        matches += 1
                
                elif 'check' in rule:
                    check_result = self._perform_check(non_empty_values, rule['check'])
                    if check_result:
                        type_score += check_result * rule['weight']
                        matches += 1
            
            if matches > 0:
                scores[data_type] = type_score / matches
        
        # Get best match
        if scores:
            best_type = max(scores, key=scores.get)
            confidence = min(scores[best_type], 1.0)
            
            return {
                'type': best_type,
                'confidence': confidence,
                'reasoning': f"Matched {best_type} patterns with {confidence:.2f} confidence",
                'scores': scores
            }
        
        # Fallback classification based on data characteristics
        fallback_type = self._fallback_classification(non_empty_values)
        
        return {
            'type': fallback_type,
            'confidence': 0.3,
            'reasoning': 'No strong pattern match, using fallback classification'
        }
    
    def _check_pattern(self, values: List[str], pattern: str, check_type: str = 'match') -> float:
        """Check how many values match a pattern"""
        matches = 0
        
        try:
            compiled_pattern = re.compile(pattern, re.IGNORECASE)
            
            for value in values:
                if check_type == 'match':
                    if compiled_pattern.match(value):
                        matches += 1
                elif check_type == 'contains':
                    if compiled_pattern.search(value):
                        matches += 1
        except:
            return 0
        
        return matches / len(values) if values else 0
    
    def _check_column_name(self, column_name: str, keywords: List[str]) -> bool:
        """Check if column name contains any keywords"""
        name_lower = column_name.lower()
        
        for keyword in keywords:
            if keyword in name_lower:
                return True
        
        return False
    
    def _check_values(self, values: List[str], expected_values: List[str]) -> float:
        """Check how many values match expected values"""
        expected_set = set(expected_values)
        matches = sum(1 for v in values if v in expected_set)
        
        return matches / len(values) if values else 0
    
    def _perform_check(self, values: List[str], check_type: str) -> float:
        """Perform specific data checks"""
        
        if check_type == 'all_numeric':
            numeric_count = sum(1 for v in values if v.replace('.', '').replace('-', '').isdigit())
            return numeric_count / len(values) if values else 0
        
        elif check_type == 'numeric_with_decimals':
            decimal_count = sum(1 for v in values if '.' in v and 
                              v.replace('.', '').replace('-', '').isdigit())
            return decimal_count / len(values) if values else 0
        
        elif check_type == 'range_0_100':
            in_range = 0
            for v in values:
                try:
                    num = float(v.replace('%', ''))
                    if 0 <= num <= 100:
                        in_range += 1
                except:
                    pass
            return in_range / len(values) if values else 0
        
        return 0
    
    def _fallback_classification(self, values: List[str]) -> str:
        """Fallback classification based on data characteristics"""
        
        # Check if mostly numeric
        numeric_count = sum(1 for v in values if v.replace('.', '').replace('-', '').isdigit())
        if numeric_count / len(values) > 0.8:
            # Check if has decimals
            decimal_count = sum(1 for v in values if '.' in v)
            if decimal_count / len(values) > 0.5:
                return 'amount'
            else:
                return 'numeric_id'
        
        # Check average length
        avg_length = np.mean([len(v) for v in values])
        
        if avg_length < 10:
            return 'category'
        elif avg_length < 50:
            return 'text_id'
        else:
            return 'description'

class LabelingFunctionGenerator:
    """Generate labeling functions for weak supervision"""
    
    def __init__(self):
        self.functions = []
        
    def generate_function(self, column_name: str, values: List[Any]) -> callable:
        """Generate a labeling function based on column characteristics"""
        
        def labeling_function(series: pd.Series) -> int:
            # Label mappings
            labels = {
                'hostname': 0,
                'ip_address': 1,
                'email': 2,
                'timestamp': 3,
                'numeric_id': 4,
                'unknown': -1
            }
            
            # Convert to string for analysis
            str_values = series.astype(str).str.lower()
            
            # Check for patterns
            if str_values.str.match(r'^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$').mean() > 0.8:
                return labels['ip_address']
            
            if str_values.str.contains('@').mean() > 0.8:
                return labels['email']
            
            if str_values.str.match(r'^\d{4}-\d{2}-\d{2}').mean() > 0.8:
                return labels['timestamp']
            
            if str_values.str.match(r'^\d+$').mean() > 0.8:
                return labels['numeric_id']
            
            # Check column name
            if any(term in column_name.lower() for term in ['host', 'server', 'machine']):
                return labels['hostname']
            
            return labels['unknown']
        
        return labeling_function
    
    def create_labeling_functions(self, column_samples: Dict[str, List]) -> List[callable]:
        """Create multiple labeling functions"""
        functions = []
        
        for column_name, values in column_samples.items():
            func = self.generate_function(column_name, values)
            functions.append(func)
        
        return functions

class LLMClassifier:
    """Main classifier using rule-based system instead of LLM"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.rule_classifier = RuleBasedClassifier(config)
        self.label_generator = LabelingFunctionGenerator()
        self.classification_cache = {}
        
    async def classify_with_llm(self, column_name: str, values: List[Any]) -> Dict[str, Any]:
        """Classify using rule-based system (renamed for compatibility)"""
        
        # Check cache
        cache_key = f"{column_name}:{str(values[:5])}"
        if cache_key in self.classification_cache:
            return self.classification_cache[cache_key]
        
        # Classify
        result = await self.rule_classifier.classify_with_rules(column_name, values)
        
        # Cache result
        self.classification_cache[cache_key] = result
        
        return result
    
    async def generate_labeling_functions(self, column_samples: Dict[str, List]) -> List[callable]:
        """Generate labeling functions for weak supervision"""
        return self.label_generator.create_labeling_functions(column_samples)
    
    def train_weak_supervision(self, df: pd.DataFrame) -> tuple:
        """Train weak supervision model (simplified without Snorkel)"""
        
        # Generate labeling functions
        functions = self.label_generator.create_labeling_functions(
            {col: df[col].tolist() for col in df.columns[:10]}
        )
        
        # Apply labeling functions
        labels = np.zeros((len(df), len(functions)))
        
        for i, func in enumerate(functions):
            for col in df.columns:
                try:
                    labels[:, i] = func(df[col])
                except:
                    pass
        
        # Simple majority vote
        predictions = []
        confidences = []
        
        for row in labels:
            non_abstain = row[row != -1]
            if len(non_abstain) > 0:
                # Get most common label
                counts = Counter(non_abstain)
                if counts:
                    most_common = counts.most_common(1)[0]
                    predictions.append(int(most_common[0]))
                    confidences.append(most_common[1] / len(non_abstain))
                else:
                    predictions.append(-1)
                    confidences.append(0.0)
            else:
                predictions.append(-1)
                confidences.append(0.0)
        
        return np.array(predictions), np.array(confidences)