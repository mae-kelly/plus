from typing import Dict, List, Set, Any, Optional
from collections import defaultdict
import json
import logging

logger = logging.getLogger(__name__)

class DataAggregator:
    
    def __init__(self):
        self.aggregated_data = defaultdict(lambda: defaultdict(set))
        self.statistics = {
            'total_aggregations': 0,
            'unique_values': defaultdict(int)
        }
        
    def aggregate_host_data(self, hostname: str, data: Dict[str, Any]) -> Dict[str, Any]:
        aggregated = {
            'hostname': hostname,
            'raw_forms': set(),
            'attributes': defaultdict(set)
        }
        
        # Process raw forms
        if 'raw_forms' in data:
            aggregated['raw_forms'].update(data['raw_forms'])
            
        # Process associated data
        if 'associated_data' in data:
            for column, values in data['associated_data'].items():
                if 'normalized' in values:
                    for val in values['normalized']:
                        if val:
                            aggregated['attributes'][column].add(val)
                            
        # Convert sets to lists for storage
        aggregated['raw_forms'] = list(aggregated['raw_forms'])
        
        for column in aggregated['attributes']:
            values = list(aggregated['attributes'][column])
            
            # Deduplicate and clean
            unique_values = self._deduplicate_values(values)
            
            # Store aggregated values
            if len(unique_values) == 1:
                aggregated['attributes'][column] = unique_values[0]
            else:
                aggregated['attributes'][column] = unique_values[:10]  # Limit to 10
                
        self.statistics['total_aggregations'] += 1
        
        return aggregated
        
    def _deduplicate_values(self, values: List[str]) -> List[str]:
        if not values:
            return []
            
        # Remove duplicates while preserving order
        seen = set()
        unique = []
        
        for val in values:
            val_lower = str(val).lower().strip()
            
            if val_lower not in seen and val_lower != 'none' and val_lower != 'null':
                seen.add(val_lower)
                unique.append(val)
                
        return unique
        
    def merge_host_occurrences(self, occurrences: List[Dict]) -> Dict[str, Any]:
        merged = {
            'tables': set(),
            'columns': set(),
            'confidence': 0.0,
            'total_occurrences': len(occurrences)
        }
        
        confidences = []
        
        for occurrence in occurrences:
            if 'table' in occurrence:
                merged['tables'].add(occurrence['table'])
            if 'column' in occurrence:
                merged['columns'].add(occurrence['column'])
            if 'confidence' in occurrence:
                confidences.append(occurrence['confidence'])
                
        # Calculate aggregate confidence
        if confidences:
            merged['confidence'] = max(confidences)
            merged['avg_confidence'] = sum(confidences) / len(confidences)
            
        # Convert sets to lists
        merged['tables'] = list(merged['tables'])
        merged['columns'] = list(merged['columns'])
        
        return merged
        
    def consolidate_attributes(self, attributes: Dict[str, Any]) -> Dict[str, Any]:
        consolidated = {}
        
        for key, value in attributes.items():
            if value is None:
                continue
                
            # Handle different value types
            if isinstance(value, set):
                value = list(value)
                
            if isinstance(value, list):
                # Remove empty values
                value = [v for v in value if v and str(v).strip()]
                
                if not value:
                    continue
                elif len(value) == 1:
                    consolidated[key] = value[0]
                else:
                    # Keep unique values
                    unique = self._deduplicate_values(value)
                    if unique:
                        consolidated[key] = unique[:5]  # Limit to 5
            else:
                # Single value
                if value and str(value).strip():
                    consolidated[key] = value
                    
        return consolidated
        
    def calculate_importance_scores(self, column_frequencies: Dict[str, int]) -> Dict[str, float]:
        if not column_frequencies:
            return {}
            
        total = sum(column_frequencies.values())
        
        importance_scores = {}
        
        for column, count in column_frequencies.items():
            # Basic importance score
            frequency_score = count / total
            
            # Boost score for certain column types
            boost = 1.0
            col_lower = column.lower()
            
            # High importance columns
            if any(term in col_lower for term in ['host', 'ip', 'domain', 'owner', 'cio']):
                boost = 2.0
            # Medium importance
            elif any(term in col_lower for term in ['env', 'location', 'app', 'class']):
                boost = 1.5
                
            importance_scores[column] = frequency_score * boost
            
        # Normalize scores
        max_score = max(importance_scores.values()) if importance_scores else 1
        
        for column in importance_scores:
            importance_scores[column] = importance_scores[column] / max_score
            
        return importance_scores
        
    def get_statistics(self) -> Dict[str, Any]:
        return {
            'total_aggregations': self.statistics['total_aggregations'],
            'unique_values_per_column': dict(self.statistics['unique_values'])
        }