import torch
import numpy as np
from typing import Dict, List, Any
from datetime import datetime
import re

class ContextAnalyzer:
    def __init__(self, device: str = 'mps'):
        self.device = device
        self.context_patterns = self._load_context_patterns()
        self.enrichment_cache = {}
    
    async def enrich_host_data(self, hostname: str, host_data: Dict, column_metadata: Dict) -> Dict:
        enriched = {
            'hostname': hostname,
            'raw_forms': list(host_data.get('raw_forms', set()))[:20],
            'occurrence_count': len(host_data.get('occurrences', [])),
            'confidence': self._calculate_confidence(host_data),
            'discovered_at': datetime.now().isoformat()
        }
        
        attributes = host_data.get('attributes', {})
        
        for attr_name, attr_values in attributes.items():
            if attr_name in column_metadata:
                col_type = column_metadata[attr_name].get('type', 'unknown')
                
                processed_value = self._process_attribute(
                    attr_name,
                    attr_values,
                    col_type
                )
                
                enriched[attr_name] = processed_value
        
        inferred_attributes = self._infer_missing_attributes(hostname, enriched)
        enriched.update(inferred_attributes)
        
        enriched['quality_score'] = self._calculate_quality_score(enriched)
        
        return enriched
    
    def _process_attribute(self, attr_name: str, values: Any, col_type: str) -> Any:
        if isinstance(values, list):
            if not values:
                return None
            
            unique_values = []
            seen = set()
            
            for val in values:
                val_str = str(val).lower().strip()
                if val_str and val_str not in seen:
                    unique_values.append(val)
                    seen.add(val_str)
            
            if len(unique_values) == 1:
                return unique_values[0]
            elif col_type in ['timestamp', 'date']:
                return self._get_most_recent(unique_values)
            elif col_type in ['amount', 'percentage', 'numeric']:
                return self._get_median(unique_values)
            else:
                return unique_values[:5]
        else:
            return values
    
    def _infer_missing_attributes(self, hostname: str, current_attrs: Dict) -> Dict:
        inferred = {}
        
        if 'environment' not in current_attrs:
            env = self._infer_environment(hostname)
            if env:
                inferred['environment_inferred'] = env
        
        if 'datacenter' not in current_attrs:
            dc = self._infer_datacenter(hostname)
            if dc:
                inferred['datacenter_inferred'] = dc
        
        if 'os_type' not in current_attrs:
            os = self._infer_os_type(hostname, current_attrs)
            if os:
                inferred['os_type_inferred'] = os
        
        if self._is_ip_address(hostname) and 'network_segment' not in current_attrs:
            segment = self._get_network_segment(hostname)
            inferred['network_segment'] = segment
        
        return inferred
    
    def _calculate_confidence(self, host_data: Dict) -> float:
        occurrences = host_data.get('occurrences', [])
        
        if not occurrences:
            return 0.0
        
        confidences = [occ.get('confidence', 0) for occ in occurrences]
        
        max_conf = max(confidences)
        avg_conf = np.mean(confidences)
        
        occurrence_factor = min(len(occurrences) / 10, 1.0)
        
        table_diversity = len(set(occ.get('table', '') for occ in occurrences))
        diversity_factor = min(table_diversity / 5, 1.0)
        
        final_confidence = (
            max_conf * 0.4 +
            avg_conf * 0.3 +
            occurrence_factor * 0.2 +
            diversity_factor * 0.1
        )
        
        return min(final_confidence, 1.0)
    
    def _calculate_quality_score(self, enriched_data: Dict) -> float:
        score = 0.0
        max_score = 0.0
        
        important_fields = {
            'hostname': 1.0,
            'environment': 0.8,
            'datacenter': 0.7,
            'os_type': 0.6,
            'owner': 0.5,
            'application': 0.5
        }
        
        for field, weight in important_fields.items():
            max_score += weight
            
            if field in enriched_data or f"{field}_inferred" in enriched_data:
                value = enriched_data.get(field) or enriched_data.get(f"{field}_inferred")
                if value and str(value).strip():
                    score += weight
        
        completeness_score = score / max_score if max_score > 0 else 0
        
        confidence = enriched_data.get('confidence', 0)
        
        occurrence_count = enriched_data.get('occurrence_count', 0)
        occurrence_score = min(occurrence_count / 10, 1.0)
        
        quality_score = (
            completeness_score * 0.5 +
            confidence * 0.3 +
            occurrence_score * 0.2
        )
        
        return quality_score
    
    def _infer_environment(self, hostname: str) -> str:
        hostname_lower = hostname.lower()
        
        patterns = {
            'production': ['prod', 'prd', 'production'],
            'staging': ['stage', 'stg', 'staging'],
            'development': ['dev', 'develop', 'development'],
            'testing': ['test', 'tst', 'qa', 'uat'],
            'sandbox': ['sandbox', 'sbx', 'demo']
        }
        
        for env, keywords in patterns.items():
            for keyword in keywords:
                if keyword in hostname_lower:
                    return env
        
        return None
    
    def _infer_datacenter(self, hostname: str) -> str:
        hostname_lower = hostname.lower()
        
        patterns = {
            'us-east-1': ['use1', 'useast1', 'virginia', 'iad'],
            'us-west-1': ['usw1', 'uswest1', 'california', 'sjc'],
            'us-west-2': ['usw2', 'uswest2', 'oregon', 'pdx'],
            'eu-west-1': ['euw1', 'euwest1', 'ireland', 'dub'],
            'ap-southeast-1': ['apse1', 'singapore', 'sin']
        }
        
        for dc, keywords in patterns.items():
            for keyword in keywords:
                if keyword in hostname_lower:
                    return dc
        
        return None
    
    def _infer_os_type(self, hostname: str, attributes: Dict) -> str:
        hostname_lower = hostname.lower()
        
        if any(term in hostname_lower for term in ['win', 'windows', 'ws']):
            return 'windows'
        elif any(term in hostname_lower for term in ['linux', 'ubuntu', 'centos', 'rhel']):
            return 'linux'
        elif any(term in hostname_lower for term in ['osx', 'mac', 'darwin']):
            return 'macos'
        
        for attr_value in attributes.values():
            if isinstance(attr_value, str):
                value_lower = attr_value.lower()
                if 'windows' in value_lower:
                    return 'windows'
                elif any(term in value_lower for term in ['linux', 'ubuntu', 'centos']):
                    return 'linux'
        
        return None
    
    def _is_ip_address(self, value: str) -> bool:
        import re
        ipv4_pattern = r'^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$'
        return bool(re.match(ipv4_pattern, value))
    
    def _get_network_segment(self, ip: str) -> str:
        parts = ip.split('.')
        if len(parts) >= 3:
            return f"{parts[0]}.{parts[1]}.{parts[2]}.0/24"
        return None
    
    def _get_most_recent(self, values: List) -> Any:
        dates = []
        
        for val in values:
            try:
                if isinstance(val, str):
                    from dateutil import parser
                    parsed = parser.parse(val)
                    dates.append((parsed, val))
            except:
                pass
        
        if dates:
            dates.sort(key=lambda x: x[0], reverse=True)
            return dates[0][1]
        
        return values[0] if values else None
    
    def _get_median(self, values: List) -> float:
        numbers = []
        
        for val in values:
            try:
                numbers.append(float(val))
            except:
                pass
        
        if numbers:
            return np.median(numbers)
        
        return values[0] if values else None
    
    def _load_context_patterns(self) -> Dict:
        return {
            'hostname_patterns': {
                'aws_ec2': r'^ec2-\d{1,3}-\d{1,3}-\d{1,3}-\d{1,3}',
                'gcp_instance': r'^[a-z][-a-z0-9]{0,62}$',
                'azure_vm': r'^[a-zA-Z][a-zA-Z0-9-]{1,59}$'
            },
            'attribute_patterns': {
                'email': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
                'phone': r'^(\+\d{1,3})?[\s.-]?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}$'
            }
        }