"""
Quality Analyzer - Analyzes CMDB data quality, detects anomalies, and generates insights
"""

import re
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import logging
import statistics

logger = logging.getLogger(__name__)

class QualityAnalyzer:
    """Analyzes CMDB data quality and generates insights"""
    
    def __init__(self):
        self.quality_dimensions = {
            'completeness': 'Percentage of required fields populated',
            'consistency': 'Data follows expected patterns and formats',
            'accuracy': 'Data reflects real infrastructure state',
            'uniqueness': 'No duplicate or redundant entries',
            'validity': 'Data conforms to business rules',
            'timeliness': 'Data is current and up-to-date'
        }
        
        self.required_fields = [
            'hostname', 'environment', 'datacenter', 'application', 'owner', 'os_type'
        ]
        
        self.anomaly_patterns = self._build_anomaly_patterns()
    
    def _build_anomaly_patterns(self) -> Dict:
        """Build patterns for anomaly detection"""
        return {
            'naming_conventions': {
                'patterns': [
                    r'^[a-z0-9\-\.]+$',  # Lowercase with hyphens and dots
                    r'^[A-Z0-9\-\.]+$',  # Uppercase with hyphens and dots
                ],
                'description': 'Hostname naming convention violations'
            },
            'ip_ranges': {
                'private_ranges': [
                    (r'^10\.', 'Private Class A'),
                    (r'^172\.(1[6-9]|2[0-9]|3[0-1])\.', 'Private Class B'),
                    (r'^192\.168\.', 'Private Class C'),
                ],
                'description': 'IP address range violations'
            },
            'resource_limits': {
                'cpu_cores': (1, 128),
                'memory_gb': (1, 1024),
                'disk_gb': (10, 100000),
                'description': 'Resource allocation outside normal bounds'
            },
            'age_limits': {
                'max_age_days': 1095,  # 3 years
                'min_age_days': 0,
                'description': 'Entity age outside expected range'
            }
        }
    
    def analyze(self, cmdb_data: Dict) -> Dict:
        """Comprehensive analysis of CMDB data quality"""
        entities = cmdb_data.get('entities', [])
        relationships = cmdb_data.get('relationships', [])
        
        if not entities:
            return {
                'overall_quality_score': 0.0,
                'total_entities': 0,
                'total_relationships': 0,
                'quality_dimensions': {},
                'field_coverage': {},
                'recommendations': ['No data found in CMDB']
            }
        
        # Calculate quality dimensions
        completeness_score = self._calculate_completeness(entities)
        consistency_score = self._calculate_consistency(entities)
        uniqueness_score = self._calculate_uniqueness(entities)
        validity_score = self._calculate_validity(entities)
        timeliness_score = self._calculate_timeliness(entities)
        relationship_score = self._calculate_relationship_quality(entities, relationships)
        
        # Calculate overall quality score
        dimension_scores = {
            'completeness': completeness_score,
            'consistency': consistency_score,
            'uniqueness': uniqueness_score,
            'validity': validity_score,
            'timeliness': timeliness_score,
            'relationships': relationship_score
        }
        
        overall_score = sum(dimension_scores.values()) / len(dimension_scores)
        
        # Calculate field coverage
        field_coverage = self._calculate_field_coverage(entities)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(dimension_scores, field_coverage)
        
        # Compile results
        return {
            'overall_quality_score': overall_score,
            'total_entities': len(entities),
            'total_relationships': len(relationships),
            'quality_dimensions': dimension_scores,
            'field_coverage': field_coverage,
            'entity_distribution': self._get_entity_distribution(entities),
            'environment_distribution': self._get_distribution(entities, 'environment'),
            'datacenter_distribution': self._get_distribution(entities, 'datacenter'),
            'application_distribution': self._get_distribution(entities, 'application'),
            'recommendations': recommendations
        }
    
    def detect_anomalies(self, cmdb_data: Dict) -> List[Dict]:
        """Detect anomalies in CMDB data"""
        anomalies = []
        entities = cmdb_data.get('entities', [])
        relationships = cmdb_data.get('relationships', [])
        
        if not entities:
            return []
        
        # Duplicate detection
        duplicate_anomalies = self._detect_duplicates(entities)
        anomalies.extend(duplicate_anomalies)
        
        # Naming convention violations
        naming_anomalies = self._detect_naming_violations(entities)
        anomalies.extend(naming_anomalies)
        
        # Resource allocation anomalies
        resource_anomalies = self._detect_resource_anomalies(entities)
        anomalies.extend(resource_anomalies)
        
        # Orphaned entities (no relationships)
        orphan_anomalies = self._detect_orphans(entities, relationships)
        anomalies.extend(orphan_anomalies)
        
        # Configuration drift
        drift_anomalies = self._detect_configuration_drift(entities)
        anomalies.extend(drift_anomalies)
        
        # Missing critical fields
        missing_field_anomalies = self._detect_missing_fields(entities)
        anomalies.extend(missing_field_anomalies)
        
        # Inconsistent data patterns
        pattern_anomalies = self._detect_pattern_violations(entities)
        anomalies.extend(pattern_anomalies)
        
        # Sort by severity
        severity_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
        anomalies.sort(key=lambda x: severity_order.get(x.get('severity', 'low'), 4))
        
        return anomalies
    
    def generate_insights(self, cmdb_data: Dict) -> List[str]:
        """Generate actionable insights from CMDB data"""
        insights = []
        entities = cmdb_data.get('entities', [])
        relationships = cmdb_data.get('relationships', [])
        statistics = cmdb_data.get('statistics', {})
        
        if not entities:
            return ['No data available for analysis']
        
        # Environment insights
        env_insights = self._generate_environment_insights(entities)
        insights.extend(env_insights)
        
        # Application insights
        app_insights = self._generate_application_insights(entities)
        insights.extend(app_insights)
        
        # Infrastructure insights
        infra_insights = self._generate_infrastructure_insights(entities)
        insights.extend(infra_insights)
        
        # Relationship insights
        rel_insights = self._generate_relationship_insights(entities, relationships)
        insights.extend(rel_insights)
        
        # Cost optimization insights
        cost_insights = self._generate_cost_insights(entities)
        insights.extend(cost_insights)
        
        # Security insights
        security_insights = self._generate_security_insights(entities)
        insights.extend(security_insights)
        
        # Capacity planning insights
        capacity_insights = self._generate_capacity_insights(entities)
        insights.extend(capacity_insights)
        
        return insights
    
    # Quality Calculation Methods
    
    def _calculate_completeness(self, entities: List[Dict]) -> float:
        """Calculate data completeness score"""
        if not entities:
            return 0.0
        
        completeness_scores = []
        
        for entity in entities:
            # Count populated required fields
            populated = sum(1 for field in self.required_fields 
                          if entity.get(field) and entity.get(field) != 'unknown')
            
            # Calculate entity completeness
            entity_score = populated / len(self.required_fields)
            completeness_scores.append(entity_score)
        
        return sum(completeness_scores) / len(completeness_scores)
    
    def _calculate_consistency(self, entities: List[Dict]) -> float:
        """Calculate data consistency score"""
        if not entities:
            return 0.0
        
        consistency_checks = []
        
        # Check hostname consistency
        hostname_patterns = self._analyze_naming_patterns(entities, 'hostname')
        consistency_checks.append(hostname_patterns['consistency_score'])
        
        # Check IP address consistency
        ip_consistency = self._check_ip_consistency(entities)
        consistency_checks.append(ip_consistency)
        
        # Check environment naming
        env_values = set(e.get('environment', '') for e in entities)
        expected_envs = {'production', 'staging', 'development', 'testing', 'sandbox', 'unknown'}
        env_consistency = len(env_values & expected_envs) / max(len(env_values), 1)
        consistency_checks.append(env_consistency)
        
        return sum(consistency_checks) / len(consistency_checks)
    
    def _calculate_uniqueness(self, entities: List[Dict]) -> float:
        """Calculate data uniqueness score"""
        if not entities:
            return 0.0
        
        # Check hostname uniqueness
        hostnames = [e.get('hostname', '') for e in entities]
        unique_hostnames = len(set(hostnames))
        uniqueness_score = unique_hostnames / len(hostnames) if hostnames else 0
        
        return uniqueness_score
    
    def _calculate_validity(self, entities: List[Dict]) -> float:
        """Calculate data validity score"""
        if not entities:
            return 0.0
        
        validity_scores = []
        
        for entity in entities:
            entity_validity = []
            
            # Check hostname validity
            hostname = entity.get('hostname', '')
            if hostname and re.match(r'^[a-zA-Z0-9\-\.]+$', hostname):
                entity_validity.append(1.0)
            else:
                entity_validity.append(0.0)
            
            # Check IP validity
            ip = entity.get('ip_address', '')
            if ip:
                if self._is_valid_ip(ip):
                    entity_validity.append(1.0)
                else:
                    entity_validity.append(0.0)
            
            # Check resource validity
            cpu = entity.get('cpu_cores')
            if cpu:
                if isinstance(cpu, (int, float)) and 0 < cpu <= 256:
                    entity_validity.append(1.0)
                else:
                    entity_validity.append(0.0)
            
            if entity_validity:
                validity_scores.append(sum(entity_validity) / len(entity_validity))
        
        return sum(validity_scores) / len(validity_scores) if validity_scores else 0.5
    
    def _calculate_timeliness(self, entities: List[Dict]) -> float:
        """Calculate data timeliness score"""
        if not entities:
            return 0.0
        
        timeliness_scores = []
        now = datetime.now()
        
        for entity in entities:
            # Check last_seen or last_modified
            last_update = entity.get('last_seen') or entity.get('last_modified')
            
            if last_update:
                try:
                    if isinstance(last_update, str):
                        update_time = datetime.fromisoformat(last_update.replace('Z', '+00:00'))
                    else:
                        update_time = last_update
                    
                    age_days = (now - update_time).days
                    
                    # Score based on age (1.0 for < 1 day, 0.0 for > 90 days)
                    if age_days < 1:
                        score = 1.0
                    elif age_days < 7:
                        score = 0.9
                    elif age_days < 30:
                        score = 0.7
                    elif age_days < 90:
                        score = 0.5
                    else:
                        score = 0.2
                    
                    timeliness_scores.append(score)
                except:
                    timeliness_scores.append(0.5)
            else:
                timeliness_scores.append(0.3)
        
        return sum(timeliness_scores) / len(timeliness_scores) if timeliness_scores else 0.5
    
    def _calculate_relationship_quality(self, entities: List[Dict], relationships: List[Dict]) -> float:
        """Calculate relationship quality score"""
        if not entities:
            return 0.0
        
        # Calculate relationship density
        max_relationships = len(entities) * (len(entities) - 1) / 2
        actual_relationships = len(relationships)
        
        # Ideal density is between 5-20% for most infrastructures
        density = actual_relationships / max_relationships if max_relationships > 0 else 0
        
        if density < 0.05:
            density_score = density * 20  # Scale up low density
        elif density > 0.2:
            density_score = 1.0 - (density - 0.2) * 2  # Penalize very high density
        else:
            density_score = 1.0
        
        # Check relationship confidence
        if relationships:
            avg_confidence = sum(r.get('confidence', 0.5) for r in relationships) / len(relationships)
        else:
            avg_confidence = 0.0
        
        return (density_score + avg_confidence) / 2
    
    def _calculate_field_coverage(self, entities: List[Dict]) -> Dict:
        """Calculate field coverage statistics"""
        field_coverage = defaultdict(int)
        total_entities = len(entities)
        
        if not total_entities:
            return {}
        
        # Count field presence
        for entity in entities:
            for field, value in entity.items():
                if value and value != 'unknown':
                    field_coverage[field] += 1
        
        # Convert to percentages
        coverage_percentages = {}
        for field, count in field_coverage.items():
            coverage_percentages[field] = (count / total_entities) * 100
        
        # Sort by coverage
        sorted_coverage = dict(sorted(coverage_percentages.items(), 
                                    key=lambda x: x[1], 
                                    reverse=True))
        
        return sorted_coverage
    
    # Anomaly Detection Methods
    
    def _detect_duplicates(self, entities: List[Dict]) -> List[Dict]:
        """Detect duplicate entities"""
        anomalies = []
        
        # Check for duplicate hostnames
        hostnames = [e.get('hostname', '') for e in entities]
        hostname_counts = Counter(hostnames)
        
        duplicates = [(name, count) for name, count in hostname_counts.items() if count > 1]
        
        if duplicates:
            for hostname, count in duplicates[:10]:  # Limit to first 10
                anomalies.append({
                    'type': 'duplicate_hostname',
                    'severity': 'high',
                    'description': f"Hostname '{hostname}' appears {count} times",
                    'affected_entities': [hostname],
                    'recommendation': 'Investigate and deduplicate entries'
                })
        
        # Check for duplicate IPs
        ips = [e.get('ip_address', '') for e in entities if e.get('ip_address')]
        ip_counts = Counter(ips)
        
        duplicate_ips = [(ip, count) for ip, count in ip_counts.items() if count > 1]
        
        for ip, count in duplicate_ips[:10]:
            anomalies.append({
                'type': 'duplicate_ip',
                'severity': 'medium',
                'description': f"IP address '{ip}' assigned to {count} entities",
                'affected_entities': [ip],
                'recommendation': 'Verify IP assignments and resolve conflicts'
            })
        
        return anomalies
    
    def _detect_naming_violations(self, entities: List[Dict]) -> List[Dict]:
        """Detect naming convention violations"""
        anomalies = []
        
        # Analyze naming patterns
        patterns_info = self._analyze_naming_patterns(entities, 'hostname')
        
        # Find outliers
        for entity in entities[:100]:  # Check first 100 to avoid too many anomalies
            hostname = entity.get('hostname', '')
            
            if hostname and not any(re.match(pattern, hostname) 
                                   for pattern in patterns_info['common_patterns'][:3]):
                anomalies.append({
                    'type': 'naming_violation',
                    'severity': 'low',
                    'description': f"Hostname '{hostname}' doesn't match common patterns",
                    'affected_entities': [hostname],
                    'recommendation': 'Consider standardizing hostname format'
                })
                
                if len(anomalies) >= 10:  # Limit anomalies
                    break
        
        return anomalies
    
    def _detect_resource_anomalies(self, entities: List[Dict]) -> List[Dict]:
        """Detect resource allocation anomalies"""
        anomalies = []
        
        # Collect resource statistics
        cpu_values = [e.get('cpu_cores', 0) for e in entities if e.get('cpu_cores')]
        memory_values = [e.get('memory_gb', 0) for e in entities if e.get('memory_gb')]
        
        if cpu_values:
            cpu_mean = statistics.mean(cpu_values)
            cpu_stdev = statistics.stdev(cpu_values) if len(cpu_values) > 1 else 0
            
            # Find outliers (> 3 standard deviations)
            for entity in entities:
                cpu = entity.get('cpu_cores', 0)
                if cpu and cpu_stdev > 0:
                    z_score = abs((cpu - cpu_mean) / cpu_stdev)
                    if z_score > 3:
                        anomalies.append({
                            'type': 'resource_outlier',
                            'severity': 'medium',
                            'description': f"Unusual CPU allocation: {cpu} cores (avg: {cpu_mean:.1f})",
                            'affected_entities': [entity.get('hostname', 'unknown')],
                            'recommendation': 'Verify resource allocation is intentional'
                        })
        
        # Check for over-provisioned development/test environments
        for entity in entities:
            env = entity.get('environment', '')
            cpu = entity.get('cpu_cores', 0)
            memory = entity.get('memory_gb', 0)
            
            if env in ['development', 'testing', 'sandbox']:
                if cpu > 16 or memory > 64:
                    anomalies.append({
                        'type': 'over_provisioned',
                        'severity': 'medium',
                        'description': f"{env.title()} environment with high resources: {cpu} CPU, {memory}GB RAM",
                        'affected_entities': [entity.get('hostname', 'unknown')],
                        'recommendation': 'Consider rightsizing non-production resources'
                    })
        
        return anomalies[:20]  # Limit to 20 anomalies
    
    def _detect_orphans(self, entities: List[Dict], relationships: List[Dict]) -> List[Dict]:
        """Detect orphaned entities with no relationships"""
        anomalies = []
        
        # Build set of entities with relationships
        connected_entities = set()
        for rel in relationships:
            connected_entities.add(rel.get('source'))
            connected_entities.add(rel.get('target'))
        
        # Find orphans
        orphans = []
        for entity in entities:
            hostname = entity.get('hostname')
            if hostname and hostname not in connected_entities:
                orphans.append(hostname)
        
        if len(orphans) > len(entities) * 0.3:  # More than 30% orphans
            anomalies.append({
                'type': 'high_orphan_rate',
                'severity': 'high',
                'description': f"{len(orphans)} entities ({100*len(orphans)/len(entities):.1f}%) have no relationships",
                'affected_entities': orphans[:10],
                'recommendation': 'Review relationship discovery rules'
            })
        elif orphans:
            for hostname in orphans[:10]:
                anomalies.append({
                    'type': 'orphaned_entity',
                    'severity': 'low',
                    'description': f"Entity '{hostname}' has no relationships",
                    'affected_entities': [hostname],
                    'recommendation': 'Verify entity is active and properly integrated'
                })
        
        return anomalies
    
    def _detect_configuration_drift(self, entities: List[Dict]) -> List[Dict]:
        """Detect configuration drift in similar entities"""
        anomalies = []
        
        # Group by application and environment
        groups = defaultdict(list)
        for entity in entities:
            app = entity.get('application', 'unknown')
            env = entity.get('environment', 'unknown')
            if app != 'unknown' and env != 'unknown':
                groups[f"{app}:{env}"].append(entity)
        
        # Check for drift within groups
        for group_key, members in groups.items():
            if len(members) < 2:
                continue
            
            # Check OS consistency
            os_types = set(m.get('os_type', 'unknown') for m in members)
            if len(os_types) > 1:
                anomalies.append({
                    'type': 'configuration_drift',
                    'severity': 'medium',
                    'description': f"Inconsistent OS types in {group_key}: {', '.join(os_types)}",
                    'affected_entities': [m.get('hostname') for m in members[:5]],
                    'recommendation': 'Standardize OS across similar systems'
                })
            
            # Check resource consistency
            cpu_values = [m.get('cpu_cores', 0) for m in members if m.get('cpu_cores')]
            if cpu_values and max(cpu_values) > min(cpu_values) * 4:
                anomalies.append({
                    'type': 'resource_imbalance',
                    'severity': 'medium',
                    'description': f"Large CPU variance in {group_key}: {min(cpu_values)}-{max(cpu_values)} cores",
                    'affected_entities': [m.get('hostname') for m in members[:5]],
                    'recommendation': 'Review resource allocation for consistency'
                })
        
        return anomalies[:15]
    
    def _detect_missing_fields(self, entities: List[Dict]) -> List[Dict]:
        """Detect entities with missing critical fields"""
        anomalies = []
        
        # Count missing fields
        missing_counts = defaultdict(list)
        
        for entity in entities:
            hostname = entity.get('hostname', 'unknown')
            for field in self.required_fields:
                if not entity.get(field) or entity.get(field) == 'unknown':
                    missing_counts[field].append(hostname)
        
        # Report significant missing fields
        for field, hostnames in missing_counts.items():
            percentage = (len(hostnames) / len(entities)) * 100
            
            if percentage > 50:
                anomalies.append({
                    'type': 'missing_field',
                    'severity': 'high',
                    'description': f"Field '{field}' missing in {percentage:.1f}% of entities",
                    'affected_entities': hostnames[:5],
                    'recommendation': f"Enrich data source with {field} information"
                })
            elif percentage > 20:
                anomalies.append({
                    'type': 'missing_field',
                    'severity': 'medium',
                    'description': f"Field '{field}' missing in {percentage:.1f}% of entities",
                    'affected_entities': hostnames[:5],
                    'recommendation': f"Investigate why {field} is frequently missing"
                })
        
        return anomalies
    
    def _detect_pattern_violations(self, entities: List[Dict]) -> List[Dict]:
        """Detect pattern violations in data"""
        anomalies = []
        
        # Check for test/temp entities in production
        for entity in entities:
            hostname = entity.get('hostname', '').lower()
            env = entity.get('environment', '').lower()
            
            if env == 'production':
                if any(word in hostname for word in ['test', 'temp', 'demo', 'sample']):
                    anomalies.append({
                        'type': 'test_in_production',
                        'severity': 'high',
                        'description': f"Possible test entity in production: {hostname}",
                        'affected_entities': [hostname],
                        'recommendation': 'Verify if entity should be in production'
                    })
        
        return anomalies[:10]
    
    # Insight Generation Methods
    
    def _generate_environment_insights(self, entities: List[Dict]) -> List[str]:
        """Generate environment-related insights"""
        insights = []
        
        env_dist = self._get_distribution(entities, 'environment')
        
        # Check production vs non-production ratio
        prod_count = env_dist.get('production', 0)
        non_prod_count = sum(v for k, v in env_dist.items() if k != 'production')
        
        if prod_count > 0 and non_prod_count > 0:
            ratio = non_prod_count / prod_count
            if ratio < 0.5:
                insights.append(f"Low non-production to production ratio ({ratio:.1f}:1) - Consider if testing resources are adequate")
            elif ratio > 3:
                insights.append(f"High non-production to production ratio ({ratio:.1f}:1) - Opportunity for resource optimization")
        
        # Check for missing environments
        expected_envs = {'production', 'staging', 'development'}
        missing_envs = expected_envs - set(env_dist.keys())
        if missing_envs:
            insights.append(f"Missing standard environments: {', '.join(missing_envs)}")
        
        return insights
    
    def _generate_application_insights(self, entities: List[Dict]) -> List[str]:
        """Generate application-related insights"""
        insights = []
        
        app_dist = self._get_distribution(entities, 'application')
        
        # Find dominant applications
        if app_dist:
            top_apps = sorted(app_dist.items(), key=lambda x: x[1], reverse=True)[:3]
            total = sum(app_dist.values())
            
            for app, count in top_apps:
                percentage = (count / total) * 100
                if percentage > 30:
                    insights.append(f"'{app}' represents {percentage:.1f}% of infrastructure - Single point of failure risk")
        
        # Check for unknown applications
        unknown_count = app_dist.get('unknown', 0)
        if unknown_count > len(entities) * 0.2:
            insights.append(f"{unknown_count} entities with unknown application - Improve application tagging")
        
        return insights
    
    def _generate_infrastructure_insights(self, entities: List[Dict]) -> List[str]:
        """Generate infrastructure insights"""
        insights = []
        
        # Datacenter distribution
        dc_dist = self._get_distribution(entities, 'datacenter')
        
        if len(dc_dist) == 1:
            insights.append("All infrastructure in single datacenter - Consider multi-region for resilience")
        elif len(dc_dist) > 5:
            insights.append(f"Infrastructure spread across {len(dc_dist)} datacenters - Review if consolidation is possible")
        
        # OS distribution
        os_dist = self._get_distribution(entities, 'os_type')
        
        if len(os_dist) > 3:
            insights.append(f"Managing {len(os_dist)} different OS types - Consider standardization")
        
        # Entity type distribution
        type_dist = self._get_distribution(entities, 'entity_type')
        
        vm_count = type_dist.get('virtual_machine', 0)
        container_count = type_dist.get('container', 0)
        physical_count = type_dist.get('physical_server', 0)
        
        if physical_count > vm_count + container_count:
            insights.append("High number of physical servers - Consider virtualization or containerization")
        elif container_count > vm_count * 2:
            insights.append("Container-heavy infrastructure - Ensure orchestration platform is robust")
        
        return insights
    
    def _generate_relationship_insights(self, entities: List[Dict], relationships: List[Dict]) -> List[str]:
        """Generate relationship insights"""
        insights = []
        
        if not relationships:
            insights.append("No relationships discovered - Review discovery configuration")
            return insights
        
        # Relationship density
        density = len(relationships) / (len(entities) * (len(entities) - 1) / 2) if len(entities) > 1 else 0
        
        if density < 0.01:
            insights.append("Very low relationship density - Many isolated components")
        elif density > 0.3:
            insights.append("Very high relationship density - Complex interdependencies")
        
        # Relationship types
        rel_types = Counter(r.get('type', 'unknown') for r in relationships)
        
        if 'same_application' in rel_types and rel_types['same_application'] > len(relationships) * 0.5:
            insights.append("Strong application clustering detected - Good architectural separation")
        
        return insights
    
    def _generate_cost_insights(self, entities: List[Dict]) -> List[str]:
        """Generate cost optimization insights"""
        insights = []
        
        # Over-provisioned non-production
        non_prod_high_resource = []
        for entity in entities:
            env = entity.get('environment', '')
            cpu = entity.get('cpu_cores', 0)
            memory = entity.get('memory_gb', 0)
            
            if env in ['development', 'testing'] and (cpu > 8 or memory > 32):
                non_prod_high_resource.append(entity.get('hostname'))
        
        if non_prod_high_resource:
            insights.append(f"{len(non_prod_high_resource)} non-production systems with high resources - Potential cost savings")
        
        # Idle or low-utilization detection (based on status)
        inactive_count = sum(1 for e in entities if e.get('status') != 'active')
        if inactive_count > len(entities) * 0.1:
            insights.append(f"{inactive_count} inactive systems - Review for decommissioning")
        
        return insights
    
    def _generate_security_insights(self, entities: List[Dict]) -> List[str]:
        """Generate security insights"""
        insights = []
        
        # Check for public IPs
        public_ips = []
        for entity in entities:
            ip = entity.get('ip_address', '')
            if ip and not any(ip.startswith(prefix) for prefix in ['10.', '172.', '192.168.']):
                public_ips.append(entity.get('hostname'))
        
        if public_ips:
            insights.append(f"{len(public_ips)} entities with public IPs - Verify security controls")
        
        # Check for missing owners
        no_owner = sum(1 for e in entities if not e.get('owner') or e.get('owner') == 'unknown')
        if no_owner > len(entities) * 0.2:
            insights.append(f"{no_owner} entities without clear ownership - Security and accountability risk")
        
        return insights
    
    def _generate_capacity_insights(self, entities: List[Dict]) -> List[str]:
        """Generate capacity planning insights"""
        insights = []
        
        # Resource utilization patterns
        cpu_values = [e.get('cpu_cores', 0) for e in entities if e.get('cpu_cores')]
        memory_values = [e.get('memory_gb', 0) for e in entities if e.get('memory_gb')]
        
        if cpu_values:
            total_cpu = sum(cpu_values)
            avg_cpu = statistics.mean(cpu_values)
            
            insights.append(f"Total CPU capacity: {total_cpu} cores across {len(cpu_values)} systems (avg: {avg_cpu:.1f})")
            
            # Check for imbalance
            if max(cpu_values) > avg_cpu * 5:
                insights.append("Large variance in CPU allocation - Consider load balancing")
        
        if memory_values:
            total_memory = sum(memory_values)
            insights.append(f"Total memory capacity: {total_memory:,} GB across {len(memory_values)} systems")
        
        return insights
    
    # Helper Methods
    
    def _get_distribution(self, entities: List[Dict], field: str) -> Dict[str, int]:
        """Get distribution of values for a field"""
        distribution = defaultdict(int)
        
        for entity in entities:
            value = entity.get(field, 'unknown')
            if value:
                distribution[value] += 1
        
        return dict(distribution)
    
    def _get_entity_distribution(self, entities: List[Dict]) -> Dict[str, int]:
        """Get distribution of entity types"""
        return self._get_distribution(entities, 'entity_type')
    
    def _analyze_naming_patterns(self, entities: List[Dict], field: str) -> Dict:
        """Analyze naming patterns in a field"""
        values = [e.get(field, '') for e in entities if e.get(field)]
        
        if not values:
            return {'common_patterns': [], 'consistency_score': 0}
        
        # Extract patterns
        patterns = defaultdict(int)
        
        for value in values:
            # Simple pattern extraction
            pattern = re.sub(r'\d+', '#', value)  # Replace numbers with #
            pattern = re.sub(r'[a-f0-9]{8,}', 'HEX', pattern)  # Replace hex strings
            patterns[pattern] += 1
        
        # Find most common patterns
        common_patterns = sorted(patterns.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Calculate consistency
        if common_patterns:
            top_pattern_coverage = common_patterns[0][1] / len(values)
            consistency_score = min(top_pattern_coverage * 2, 1.0)  # Scale up
        else:
            consistency_score = 0
        
        return {
            'common_patterns': [p[0] for p in common_patterns],
            'consistency_score': consistency_score
        }
    
    def _check_ip_consistency(self, entities: List[Dict]) -> float:
        """Check IP address consistency"""
        ips = [e.get('ip_address', '') for e in entities if e.get('ip_address')]
        
        if not ips:
            return 0.5
        
        valid_ips = sum(1 for ip in ips if self._is_valid_ip(ip))
        
        return valid_ips / len(ips)
    
    def _is_valid_ip(self, ip: str) -> bool:
        """Validate IP address format"""
        parts = ip.split('.')
        if len(parts) != 4:
            return False
        
        try:
            return all(0 <= int(part) <= 255 for part in parts)
        except ValueError:
            return False
    
    def _generate_recommendations(self, dimension_scores: Dict, field_coverage: Dict) -> List[str]:
        """Generate recommendations based on quality analysis"""
        recommendations = []
        
        # Check dimension scores
        for dimension, score in dimension_scores.items():
            if score < 0.5:
                if dimension == 'completeness':
                    recommendations.append("Enrich data sources to improve completeness")
                elif dimension == 'consistency':
                    recommendations.append("Standardize naming conventions and data formats")
                elif dimension == 'uniqueness':
                    recommendations.append("Implement deduplication process")
                elif dimension == 'validity':
                    recommendations.append("Add data validation rules to discovery process")
                elif dimension == 'timeliness':
                    recommendations.append("Schedule regular discovery runs to keep data current")
                elif dimension == 'relationships':
                    recommendations.append("Review relationship mapping rules")
        
        # Check field coverage
        critical_fields = ['environment', 'owner', 'application', 'datacenter']
        for field in critical_fields:
            coverage = field_coverage.get(field, 0)
            if coverage < 50:
                recommendations.append(f"Improve {field} identification - only {coverage:.1f}% coverage")
        
        return recommendations[:10]  # Limit to top 10 recommendations