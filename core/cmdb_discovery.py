"""
CMDB Discovery - Discovers hosts and infrastructure from BigQuery data
"""

import re
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class CMDBDiscovery:
    """Discovers infrastructure components from BigQuery data"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.discovery_config = config.get('discovery', {})
        
        # Pattern configurations
        self.hostname_patterns = self.discovery_config.get('hostname_patterns', [])
        self.ip_patterns = self.discovery_config.get('ip_patterns', [])
        self.env_patterns = self.discovery_config.get('environment_patterns', [])
        self.app_patterns = self.discovery_config.get('application_patterns', [])
        
        # Discovery results
        self.discovered_hosts = {}
        self.column_mappings = {}
        self.statistics = defaultdict(int)
    
    async def discover_hosts(self, bigquery_data: List[Dict]) -> Dict[str, Dict]:
        """Discover hosts from BigQuery data"""
        logger.info("Starting host discovery from BigQuery data...")
        
        for data_source in bigquery_data:
            source_name = data_source.get('source', 'unknown')
            
            for table_data in data_source.get('tables', []):
                await self._process_table(table_data, source_name)
        
        # Enrich discovered hosts
        self._enrich_hosts()
        
        # Deduplicate and merge
        self._deduplicate_hosts()
        
        logger.info(f"Discovered {len(self.discovered_hosts)} unique hosts")
        return self.discovered_hosts
    
    async def _process_table(self, table_data: Dict, source_name: str):
        """Process a BigQuery table to find hosts"""
        table_name = table_data.get('full_name', table_data.get('name', 'unknown'))
        rows = table_data.get('rows', [])
        columns = table_data.get('columns', {})
        
        if not rows:
            return
        
        logger.debug(f"Processing table {table_name} with {len(rows)} rows")
        self.statistics['tables_processed'] += 1
        self.statistics['rows_processed'] += len(rows)
        
        # Identify key columns
        hostname_columns = self._identify_hostname_columns(columns)
        ip_columns = self._identify_ip_columns(columns)
        env_columns = self._identify_env_columns(columns)
        app_columns = self._identify_app_columns(columns)
        
        # Store column mappings
        self.column_mappings[table_name] = {
            'hostname_columns': hostname_columns,
            'ip_columns': ip_columns,
            'env_columns': env_columns,
            'app_columns': app_columns
        }
        
        # Process each row
        for row in rows:
            self._process_row(row, table_name, source_name, 
                            hostname_columns, ip_columns, 
                            env_columns, app_columns)
    
    def _identify_hostname_columns(self, columns: Dict) -> List[Tuple[str, float]]:
        """Identify columns likely to contain hostnames"""
        hostname_cols = []
        
        for col_name, col_info in columns.items():
            confidence = 0.0
            col_lower = col_name.lower()
            
            # Check against configured patterns
            for pattern_config in self.hostname_patterns:
                pattern = pattern_config.get('column_pattern', '')
                if re.match(pattern, col_lower):
                    confidence = max(confidence, pattern_config.get('confidence', 0.5))
            
            # Check semantic type inference
            if col_info.get('potential_type') == 'hostname':
                confidence = max(confidence, 0.7)
            
            # Check statistics
            stats = col_info.get('statistics', {})
            if stats.get('hostname_likelihood', 0) > 0.6:
                confidence = max(confidence, stats['hostname_likelihood'])
            
            # High uniqueness is good for hostnames
            if stats.get('unique_ratio', 0) > 0.9:
                confidence += 0.1
            
            if confidence > 0.5:
                hostname_cols.append((col_name, confidence))
                logger.debug(f"  Identified hostname column: {col_name} (confidence: {confidence:.2f})")
        
        return sorted(hostname_cols, key=lambda x: x[1], reverse=True)
    
    def _identify_ip_columns(self, columns: Dict) -> List[Tuple[str, float]]:
        """Identify columns likely to contain IP addresses"""
        ip_cols = []
        
        for col_name, col_info in columns.items():
            confidence = 0.0
            col_lower = col_name.lower()
            
            # Check against configured patterns
            for pattern_config in self.ip_patterns:
                pattern = pattern_config.get('column_pattern', '')
                if re.match(pattern, col_lower):
                    confidence = max(confidence, pattern_config.get('confidence', 0.5))
            
            # Check semantic type
            if col_info.get('potential_type') == 'ip_address':
                confidence = max(confidence, 0.8)
            
            # Check statistics
            stats = col_info.get('statistics', {})
            if stats.get('ip_likelihood', 0) > 0.7:
                confidence = max(confidence, stats['ip_likelihood'])
            
            if confidence > 0.5:
                ip_cols.append((col_name, confidence))
                logger.debug(f"  Identified IP column: {col_name} (confidence: {confidence:.2f})")
        
        return sorted(ip_cols, key=lambda x: x[1], reverse=True)
    
    def _identify_env_columns(self, columns: Dict) -> List[str]:
        """Identify environment columns"""
        env_cols = []
        
        for col_name, col_info in columns.items():
            col_lower = col_name.lower()
            
            for pattern_config in self.env_patterns:
                pattern = pattern_config.get('column_pattern', '')
                if re.match(pattern, col_lower):
                    env_cols.append(col_name)
                    break
            
            if col_info.get('potential_type') == 'environment':
                if col_name not in env_cols:
                    env_cols.append(col_name)
        
        return env_cols
    
    def _identify_app_columns(self, columns: Dict) -> List[str]:
        """Identify application columns"""
        app_cols = []
        
        for col_name, col_info in columns.items():
            col_lower = col_name.lower()
            
            for pattern_config in self.app_patterns:
                pattern = pattern_config.get('column_pattern', '')
                if re.match(pattern, col_lower):
                    app_cols.append(col_name)
                    break
            
            if col_info.get('potential_type') == 'application':
                if col_name not in app_cols:
                    app_cols.append(col_name)
        
        return app_cols
    
    def _process_row(self, row: Dict, table_name: str, source_name: str,
                     hostname_columns: List[Tuple[str, float]], 
                     ip_columns: List[Tuple[str, float]],
                     env_columns: List[str], 
                     app_columns: List[str]):
        """Process a single row to extract host information"""
        
        # Find hostname
        hostname = None
        hostname_confidence = 0.0
        
        for col_name, confidence in hostname_columns:
            value = row.get(col_name)
            if value and self._is_valid_hostname(value):
                hostname = str(value).lower()
                hostname_confidence = confidence
                break
        
        # If no hostname, try to use IP as identifier
        if not hostname:
            for col_name, confidence in ip_columns:
                value = row.get(col_name)
                if value and self._is_valid_ip(value):
                    hostname = str(value)
                    hostname_confidence = confidence * 0.7  # Lower confidence for IP-based
                    break
        
        if not hostname:
            return
        
        # Create or update host entry
        if hostname not in self.discovered_hosts:
            self.discovered_hosts[hostname] = {
                'hostname': hostname,
                'sources': [],
                'attributes': defaultdict(list),
                'confidence': hostname_confidence,
                'discovered_at': datetime.now().isoformat(),
                'bigquery_tables': []
            }
            self.statistics['hosts_discovered'] += 1
        
        host = self.discovered_hosts[hostname]
        
        # Update source tracking
        host['sources'].append({
            'table': table_name,
            'source': source_name,
            'confidence': hostname_confidence
        })
        
        # Track BigQuery tables
        if table_name not in host['bigquery_tables']:
            host['bigquery_tables'].append(table_name)
        
        # Update confidence (keep highest)
        host['confidence'] = max(host['confidence'], hostname_confidence)
        
        # Extract IP address
        for col_name, _ in ip_columns:
            value = row.get(col_name)
            if value and self._is_valid_ip(value):
                host['attributes']['ip_address'].append(str(value))
        
        # Extract environment
        for col_name in env_columns:
            value = row.get(col_name)
            if value:
                host['attributes']['environment'].append(str(value).lower())
        
        # Extract application
        for col_name in app_columns:
            value = row.get(col_name)
            if value:
                host['attributes']['application'].append(str(value).lower())
        
        # Store all other attributes
        for col_name, value in row.items():
            if value is not None:
                # Skip already processed columns
                if col_name not in [c[0] for c in hostname_columns + ip_columns]:
                    if col_name not in env_columns + app_columns:
                        host['attributes'][col_name].append(value)
    
    def _is_valid_hostname(self, value: Any) -> bool:
        """Check if value is a valid hostname"""
        if not value:
            return False
        
        hostname = str(value).lower()
        
        # Basic hostname validation
        if len(hostname) > 253:
            return False
        
        # Check for valid hostname pattern
        hostname_regex = re.compile(
            r'^[a-z0-9]([a-z0-9\-]{0,61}[a-z0-9])?(\.[a-z0-9]([a-z0-9\-]{0,61}[a-z0-9])?)*$'
        )
        
        return bool(hostname_regex.match(hostname))
    
    def _is_valid_ip(self, value: Any) -> bool:
        """Check if value is a valid IP address"""
        if not value:
            return False
        
        ip = str(value)
        
        # IPv4 validation
        ipv4_regex = re.compile(r'^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$')
        if ipv4_regex.match(ip):
            # Check octets are valid
            octets = ip.split('.')
            return all(0 <= int(octet) <= 255 for octet in octets)
        
        # IPv6 validation (simplified)
        ipv6_regex = re.compile(r'^(?:[0-9a-fA-F]{0,4}:){7}[0-9a-fA-F]{0,4}$')
        return bool(ipv6_regex.match(ip))
    
    def _enrich_hosts(self):
        """Enrich discovered hosts with derived information"""
        for hostname, host in self.discovered_hosts.items():
            # Deduplicate and clean attributes
            for attr_name, values in host['attributes'].items():
                # Remove duplicates while preserving order
                seen = set()
                unique_values = []
                for value in values:
                    value_key = str(value).lower()
                    if value_key not in seen:
                        seen.add(value_key)
                        unique_values.append(value)
                
                # Keep most common value as primary
                if unique_values:
                    if len(unique_values) == 1:
                        host[attr_name] = unique_values[0]
                    else:
                        # Store primary and alternates
                        host[attr_name] = unique_values[0]
                        if len(unique_values) > 1:
                            host[f'{attr_name}_alternates'] = unique_values[1:]
            
            # Infer missing attributes
            if 'environment' not in host:
                host['environment'] = self._infer_environment(hostname, host)
            
            if 'datacenter' not in host:
                host['datacenter'] = self._infer_datacenter(hostname, host)
            
            if 'host_type' not in host:
                host['host_type'] = self._infer_host_type(hostname, host)
            
            # Calculate quality score
            host['quality_score'] = self._calculate_quality_score(host)
    
    def _infer_environment(self, hostname: str, host: Dict) -> str:
        """Infer environment from hostname patterns"""
        hostname_lower = hostname.lower()
        
        patterns = {
            'production': ['prod', 'prd', 'live'],
            'staging': ['stage', 'stg', 'uat'],
            'development': ['dev', 'develop'],
            'testing': ['test', 'tst', 'qa']
        }
        
        for env, keywords in patterns.items():
            for keyword in keywords:
                if keyword in hostname_lower:
                    return env
        
        return 'unknown'
    
    def _infer_datacenter(self, hostname: str, host: Dict) -> str:
        """Infer datacenter/region from hostname or IP"""
        hostname_lower = hostname.lower()
        
        # Check for region patterns
        regions = {
            'us-east-1': ['use1', 'useast', 'virginia'],
            'us-west-2': ['usw2', 'uswest', 'oregon'],
            'eu-west-1': ['euw1', 'euwest', 'ireland'],
            'asia-southeast-1': ['apse1', 'singapore']
        }
        
        for region, keywords in regions.items():
            for keyword in keywords:
                if keyword in hostname_lower:
                    return region
        
        # Check IP ranges (simplified)
        ip = host.get('ip_address')
        if ip and ip.startswith('10.'):
            octet2 = int(ip.split('.')[1])
            if octet2 < 50:
                return 'us-east-1'
            elif octet2 < 100:
                return 'us-west-2'
            elif octet2 < 150:
                return 'eu-west-1'
        
        return 'unknown'
    
    def _infer_host_type(self, hostname: str, host: Dict) -> str:
        """Infer host type from hostname patterns"""
        hostname_lower = hostname.lower()
        
        if any(kw in hostname_lower for kw in ['web', 'www', 'nginx', 'apache']):
            return 'web_server'
        elif any(kw in hostname_lower for kw in ['db', 'database', 'mysql', 'postgres', 'mongo']):
            return 'database'
        elif any(kw in hostname_lower for kw in ['api', 'rest', 'graphql']):
            return 'api_server'
        elif any(kw in hostname_lower for kw in ['cache', 'redis', 'memcache']):
            return 'cache'
        elif any(kw in hostname_lower for kw in ['queue', 'rabbit', 'kafka', 'sqs']):
            return 'message_queue'
        elif any(kw in hostname_lower for kw in ['lb', 'load', 'balancer', 'haproxy']):
            return 'load_balancer'
        
        return 'compute'
    
    def _calculate_quality_score(self, host: Dict) -> float:
        """Calculate data quality score for a host"""
        score = 0.0
        max_score = 0.0
        
        # Check important fields
        important_fields = {
            'ip_address': 0.2,
            'environment': 0.2,
            'application': 0.15,
            'datacenter': 0.15,
            'host_type': 0.1
        }
        
        for field, weight in important_fields.items():
            max_score += weight
            if host.get(field) and host.get(field) != 'unknown':
                score += weight
        
        # Confidence contributes to score
        score += host.get('confidence', 0) * 0.2
        max_score += 0.2
        
        return score / max_score
    
    def _deduplicate_hosts(self):
        """Deduplicate hosts that might be the same"""
        # Group potential duplicates
        groups = defaultdict(list)
        
        for hostname, host in self.discovered_hosts.items():
            # Group by IP if available
            ip = host.get('ip_address')
            if ip:
                groups[f'ip:{ip}'].append(hostname)
            
            # Group by common patterns
            base_name = re.sub(r'\d+', '', hostname)
            if '.' in hostname:
                base_name = hostname.split('.')[0]
            groups[f'base:{base_name}'].append(hostname)
        
        # Merge duplicates
        merged = set()
        for group_key, hostnames in groups.items():
            if len(hostnames) > 1 and not all(h in merged for h in hostnames):
                # Keep the one with highest quality score
                best_host = max(hostnames, 
                              key=lambda h: self.discovered_hosts[h].get('quality_score', 0))
                
                # Merge others into best
                for hostname in hostnames:
                    if hostname != best_host and hostname not in merged:
                        self._merge_hosts(best_host, hostname)
                        merged.add(hostname)
        
        # Remove merged hosts
        for hostname in merged:
            del self.discovered_hosts[hostname]
    
    def _merge_hosts(self, primary: str, secondary: str):
        """Merge secondary host into primary"""
        primary_host = self.discovered_hosts[primary]
        secondary_host = self.discovered_hosts[secondary]
        
        # Merge sources
        primary_host['sources'].extend(secondary_host['sources'])
        
        # Merge BigQuery tables
        for table in secondary_host.get('bigquery_tables', []):
            if table not in primary_host['bigquery_tables']:
                primary_host['bigquery_tables'].append(table)
        
        # Merge attributes
        for attr_name, values in secondary_host.get('attributes', {}).items():
            if attr_name in primary_host['attributes']:
                primary_host['attributes'][attr_name].extend(values)
            else:
                primary_host['attributes'][attr_name] = values
        
        # Update confidence
        primary_host['confidence'] = max(primary_host['confidence'], 
                                        secondary_host.get('confidence', 0))
        
        logger.debug(f"Merged {secondary} into {primary}")
    
    async def find_relationships(self, hosts: Dict[str, Dict]) -> List[Dict]:
        """Find relationships between discovered hosts"""
        relationships = []
        
        # Find relationships based on shared attributes
        for host1_name, host1 in hosts.items():
            for host2_name, host2 in hosts.items():
                if host1_name >= host2_name:  # Avoid duplicates
                    continue
                
                # Same environment
                if (host1.get('environment') == host2.get('environment') != 'unknown'):
                    rel_type = 'same_environment'
                    confidence = 0.6
                    
                    # Same application in same environment
                    if (host1.get('application') == host2.get('application') != 'unknown'):
                        rel_type = 'same_application'
                        confidence = 0.9
                    
                    relationships.append({
                        'source': host1_name,
                        'target': host2_name,
                        'type': rel_type,
                        'confidence': confidence,
                        'discovered_at': datetime.now().isoformat()
                    })
                
                # Same datacenter
                elif (host1.get('datacenter') == host2.get('datacenter') != 'unknown'):
                    relationships.append({
                        'source': host1_name,
                        'target': host2_name,
                        'type': 'same_datacenter',
                        'confidence': 0.5,
                        'discovered_at': datetime.now().isoformat()
                    })
        
        logger.info(f"Found {len(relationships)} relationships")
        return relationships