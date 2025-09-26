# core/orchestrator.py
"""
Discovery Orchestrator - Manages the entire discovery pipeline
"""

import asyncio
import json
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from collections import defaultdict
import logging

# Remove DataScanner import and use BigQueryScanner instead
from core.bigquery_scanner import BigQueryScanner
from core.cmdb_discovery import CMDBDiscovery
from core.classifier import IntelligentClassifier
from core.relationship_mapper import RelationshipMapper
from core.cmdb_builder import CMDBBuilder
from core.quality_analyzer import QualityAnalyzer

logger = logging.getLogger(__name__)

class DiscoveryOrchestrator:
    """Orchestrates the entire discovery process"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.start_time = datetime.now()
        
        # Initialize components - use BigQueryScanner instead of DataScanner
        self.scanner = BigQueryScanner(config)
        self.discovery = CMDBDiscovery(config)
        self.classifier = IntelligentClassifier()
        self.relationship_mapper = RelationshipMapper()
        self.cmdb_builder = CMDBBuilder(config.get('output_database', 'cmdb.db'))
        self.quality_analyzer = QualityAnalyzer()
        
        # State tracking
        self.state = {
            'phase': 'initialized',
            'discovered_hosts': {},
            'classified_entities': {},
            'relationships': [],
            'statistics': defaultdict(int)
        }
        
        self.checkpoint_file = Path('checkpoints/discovery_state.pkl')
        self.checkpoint_file.parent.mkdir(exist_ok=True)
    
    async def scan_data_sources(self) -> List[Dict]:
        """Scan all configured BigQuery data sources"""
        self.state['phase'] = 'scanning'
        discovered_data = []
        
        # Scan BigQuery projects
        logger.info("Scanning BigQuery projects...")
        bigquery_data = await self.scanner.scan_all_projects()
        discovered_data.extend(bigquery_data)
        logger.info(f"Scanned {len(bigquery_data)} BigQuery sources")
        
        self.state['statistics']['data_sources'] = len(discovered_data)
        return discovered_data
    
    async def extract_hosts(self, data_sources: List[Dict]) -> Dict[str, Dict]:
        """Extract potential hosts from data sources"""
        self.state['phase'] = 'extracting'
        
        # Use CMDBDiscovery to discover hosts
        hosts = await self.discovery.discover_hosts(data_sources)
        
        self.state['discovered_hosts'] = hosts
        self.state['statistics']['hosts_discovered'] = len(hosts)
        
        logger.info(f"Extracted {len(hosts)} unique hosts")
        return hosts
    
    async def classify_entities(self, hosts: Dict[str, Dict]) -> Dict[str, Dict]:
        """Classify and enrich host entities"""
        self.state['phase'] = 'classifying'
        classified = {}
        
        for hostname, host_data in hosts.items():
            # Classify the host
            classification = await self.classifier.classify_host(hostname, host_data)
            
            # Enrich with classification
            enriched_data = {
                **host_data,
                'classification': classification,
                'entity_type': classification['type'],
                'sub_type': classification.get('sub_type'),
                'criticality': self._calculate_criticality(hostname, host_data, classification)
            }
            
            classified[hostname] = enriched_data
        
        self.state['classified_entities'] = classified
        return classified
    
    def _calculate_criticality(self, hostname: str, host_data: Dict, classification: Dict) -> str:
        """Calculate host criticality level"""
        score = 0
        
        # Environment factor
        env = host_data.get('environment', 'unknown')
        if env == 'production':
            score += 40
        elif env == 'disaster_recovery':
            score += 35
        elif env == 'staging':
            score += 20
        elif env == 'development':
            score += 10
        
        # Application type factor
        app = host_data.get('application', 'unknown')
        critical_apps = ['database', 'api', 'load_balancer', 'queue', 'cache']
        if app in critical_apps:
            score += 30
        elif app in ['web', 'mail']:
            score += 20
        elif app in ['monitoring', 'logging']:
            score += 15
        
        # Classification confidence
        if classification.get('confidence', 0) > 0.9:
            score += 10
        
        # Number of sources (indicates importance)
        n_sources = len(host_data.get('sources', []))
        if n_sources > 5:
            score += 20
        elif n_sources > 2:
            score += 10
        
        # Determine criticality level
        if score >= 70:
            return 'critical'
        elif score >= 50:
            return 'high'
        elif score >= 30:
            return 'medium'
        else:
            return 'low'
    
    async def map_relationships(self, entities: Dict[str, Dict]) -> List[Dict]:
        """Map relationships between entities"""
        self.state['phase'] = 'mapping_relationships'
        
        relationships = self.relationship_mapper.map_relationships(entities)
        
        # Also find relationships from CMDBDiscovery
        discovery_relationships = await self.discovery.find_relationships(entities)
        relationships.extend(discovery_relationships)
        
        # Deduplicate
        unique_rels = {}
        for rel in relationships:
            key = (rel['source'], rel['target'], rel['type'])
            if key not in unique_rels or unique_rels[key]['confidence'] < rel['confidence']:
                unique_rels[key] = rel
        
        relationships = list(unique_rels.values())
        
        self.state['relationships'] = relationships
        self.state['statistics']['relationships'] = len(relationships)
        
        logger.info(f"Mapped {len(relationships)} relationships")
        return relationships
    
    async def build_cmdb(self, entities: Dict[str, Dict], relationships: List[Dict]):
        """Build the CMDB database"""
        self.state['phase'] = 'building_cmdb'
        
        # Initialize database
        await self.cmdb_builder.initialize()
        
        # Create schema based on discovered attributes
        all_attributes = set()
        for entity in entities.values():
            all_attributes.update(entity.get('attributes', {}).keys())
        
        await self.cmdb_builder.create_schema(list(all_attributes))
        
        # Insert entities
        await self.cmdb_builder.insert_entities(list(entities.values()))
        
        # Insert relationships
        await self.cmdb_builder.insert_relationships(relationships)
        
        # Create indexes for performance
        await self.cmdb_builder.create_indexes()
        
        # Calculate and store statistics
        stats = self.calculate_statistics(entities, relationships)
        await self.cmdb_builder.store_statistics(stats)
        
        self.state['phase'] = 'completed'
        self.state['end_time'] = datetime.now().isoformat()
        
        logger.info(f"CMDB built successfully with {len(entities)} entities and {len(relationships)} relationships")
    
    def calculate_statistics(self, entities: Dict[str, Dict], relationships: List[Dict]) -> Dict:
        """Calculate comprehensive statistics"""
        stats = {
            'start_time': self.start_time.isoformat(),
            'end_time': datetime.now().isoformat(),
            'duration': str(datetime.now() - self.start_time),
            **self.state['statistics']
        }
        
        # Entity statistics
        stats['total_entities'] = len(entities)
        stats['environments'] = len(set(e.get('environment') for e in entities.values()))
        stats['datacenters'] = len(set(e.get('datacenter') for e in entities.values()))
        stats['applications'] = len(set(e.get('application') for e in entities.values()))
        stats['os_types'] = len(set(e.get('os_type') for e in entities.values()))
        
        # Criticality distribution
        criticality_dist = defaultdict(int)
        for entity in entities.values():
            criticality_dist[entity.get('criticality', 'unknown')] += 1
        stats['criticality_distribution'] = dict(criticality_dist)
        
        # Relationship statistics
        stats['total_relationships'] = len(relationships)
        rel_type_dist = defaultdict(int)
        for rel in relationships:
            rel_type_dist[rel['type']] += 1
        stats['relationship_types'] = dict(rel_type_dist)
        
        # Quality metrics
        entities_with_env = sum(1 for e in entities.values() if e.get('environment') != 'unknown')
        entities_with_app = sum(1 for e in entities.values() if e.get('application') != 'unknown')
        entities_with_owner = sum(1 for e in entities.values() if e.get('owner') != 'unknown')
        
        stats['classification_accuracy'] = (entities_with_env + entities_with_app + entities_with_owner) / (3 * len(entities)) if entities else 0
        
        # Average confidence scores
        avg_entity_confidence = sum(e.get('confidence', 0) for e in entities.values()) / len(entities) if entities else 0
        avg_rel_confidence = sum(r.get('confidence', 0) for r in relationships) / len(relationships) if relationships else 0
        
        stats['avg_entity_confidence'] = avg_entity_confidence
        stats['avg_relationship_confidence'] = avg_rel_confidence
        stats['quality_score'] = (avg_entity_confidence + stats['classification_accuracy']) / 2
        
        # BigQuery specific stats
        stats['projects_scanned'] = len(self.scanner.projects_scanned)
        stats['datasets_scanned'] = self.scanner.datasets_scanned
        stats['tables_scanned'] = self.scanner.tables_scanned
        stats['rows_processed'] = self.scanner.rows_processed
        
        return stats
    
    async def load_cmdb(self) -> Dict:
        """Load existing CMDB data"""
        return await self.cmdb_builder.load_all_data()
    
    async def analyze_data_quality(self, cmdb_data: Dict) -> Dict:
        """Analyze CMDB data quality"""
        return self.quality_analyzer.analyze(cmdb_data)
    
    async def detect_anomalies(self, cmdb_data: Dict) -> List[Dict]:
        """Detect anomalies in CMDB data"""
        return self.quality_analyzer.detect_anomalies(cmdb_data)
    
    async def generate_insights(self, cmdb_data: Dict) -> List[str]:
        """Generate insights from CMDB data"""
        return self.quality_analyzer.generate_insights(cmdb_data)
    
    def save_checkpoint(self):
        """Save current state to checkpoint file"""
        try:
            with open(self.checkpoint_file, 'wb') as f:
                pickle.dump(self.state, f)
            logger.info(f"Checkpoint saved to {self.checkpoint_file}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
    
    def load_checkpoint(self) -> bool:
        """Load state from checkpoint file"""
        if not self.checkpoint_file.exists():
            return False
        
        try:
            with open(self.checkpoint_file, 'rb') as f:
                self.state = pickle.load(f)
            logger.info(f"Checkpoint loaded from {self.checkpoint_file}")
            logger.info(f"Resuming from phase: {self.state.get('phase')}")
            return True
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return False
    
    def get_statistics(self) -> Dict:
        """Get current statistics"""
        return self.calculate_statistics(
            self.state.get('classified_entities', {}),
            self.state.get('relationships', [])
        )
    
    async def run_discovery_pipeline(self) -> Dict:
        """Run the complete discovery pipeline"""
        try:
            # Check for checkpoint
            if self.config.get('checkpoint', {}).get('enabled', True):
                if self.load_checkpoint():
                    logger.info("Resuming from checkpoint...")
            
            # Step 1: Scan data sources
            if self.state['phase'] in ['initialized', 'scanning']:
                data_sources = await self.scan_data_sources()
                self.save_checkpoint()
            else:
                data_sources = []  # Already scanned
            
            # Step 2: Extract hosts
            if self.state['phase'] in ['initialized', 'scanning', 'extracting']:
                if not self.state.get('discovered_hosts'):
                    hosts = await self.extract_hosts(data_sources)
                else:
                    hosts = self.state['discovered_hosts']
                self.save_checkpoint()
            else:
                hosts = self.state.get('discovered_hosts', {})
            
            # Step 3: Classify entities
            if self.state['phase'] in ['initialized', 'scanning', 'extracting', 'classifying']:
                if not self.state.get('classified_entities'):
                    entities = await self.classify_entities(hosts)
                else:
                    entities = self.state['classified_entities']
                self.save_checkpoint()
            else:
                entities = self.state.get('classified_entities', {})
            
            # Step 4: Map relationships
            if self.state['phase'] in ['initialized', 'scanning', 'extracting', 'classifying', 'mapping_relationships']:
                if not self.state.get('relationships'):
                    relationships = await self.map_relationships(entities)
                else:
                    relationships = self.state['relationships']
                self.save_checkpoint()
            else:
                relationships = self.state.get('relationships', [])
            
            # Step 5: Build CMDB
            if self.state['phase'] != 'completed':
                await self.build_cmdb(entities, relationships)
                self.save_checkpoint()
            
            # Return final results
            return {
                'entities': entities,
                'relationships': relationships,
                'statistics': self.get_statistics()
            }
            
        except Exception as e:
            logger.error(f"Discovery pipeline failed: {e}")
            self.save_checkpoint()  # Save state even on failure
            raise