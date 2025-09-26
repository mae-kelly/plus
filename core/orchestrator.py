import asyncio
import logging
from typing import Dict, List, Any
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from core.bigquery_scanner import BigQueryScanner
from core.checkpoint_manager import CheckpointManager
from models.ensemble_predictor import EnsemblePredictor
from processors.feature_extractor import AdvancedFeatureExtractor
from processors.entity_resolver import EntityResolver
from processors.graph_builder import GraphBuilder
from processors.context_analyzer import ContextAnalyzer
from storage.cmdb_builder import CMDBBuilder
from storage.metadata_store import MetadataStore
from storage.streaming_handler import StreamingHandler

logger = logging.getLogger(__name__)

class AdvancedOrchestrator:
    def __init__(self, config: Dict, device: str):
        self.config = config
        self.device = device
        self.start_time = datetime.now()
        
        self.scanner = BigQueryScanner(config)
        self.checkpoint = CheckpointManager()
        self.ensemble = EnsemblePredictor(config, device)
        self.feature_extractor = AdvancedFeatureExtractor(device)
        self.entity_resolver = EntityResolver(config, device)
        self.graph_builder = GraphBuilder()
        self.context_analyzer = ContextAnalyzer(device)
        self.cmdb = CMDBBuilder(config['storage']['duckdb_path'])
        self.metadata_store = MetadataStore(config)
        self.streaming = StreamingHandler(config) if config['storage']['streaming_enabled'] else None
        
        self.executor = ThreadPoolExecutor(max_workers=config['max_workers'])
        self.discovered_hosts = {}
        self.column_metadata = {}
        self.statistics = {
            'tables_processed': 0,
            'rows_scanned': 0,
            'hosts_discovered': 0,
            'columns_classified': 0,
            'entities_resolved': 0
        }
    
    async def execute(self):
        logger.info("Starting advanced CMDB discovery")
        
        state = self.checkpoint.load()
        if state:
            self.discovered_hosts = state.get('hosts', {})
            self.column_metadata = state.get('metadata', {})
            self.statistics = state.get('statistics', self.statistics)
            logger.info(f"Resumed from checkpoint: {self.statistics['tables_processed']} tables")
        
        await self._phase1_discovery()
        await self._phase2_classification()
        await self._phase3_resolution()
        await self._phase4_build()
        
        self._report_statistics()
    
    async def _phase1_discovery(self):
        logger.info("Phase 1: Data discovery and scanning")
        
        async for batch in self.scanner.scan_projects(self.config['projects']):
            features = await self._extract_features(batch)
            
            for table_data in batch:
                self.statistics['tables_processed'] += 1
                self.statistics['rows_scanned'] += len(table_data['rows'])
                
                await self._process_table(table_data, features)
                
                if self.statistics['tables_processed'] % 100 == 0:
                    self.checkpoint.save({
                        'hosts': self.discovered_hosts,
                        'metadata': self.column_metadata,
                        'statistics': self.statistics
                    })
    
    async def _phase2_classification(self):
        logger.info("Phase 2: Advanced column classification")
        
        column_predictions = await self.ensemble.classify_columns(
            self.column_metadata,
            self.discovered_hosts
        )
        
        for column_name, prediction in column_predictions.items():
            self.column_metadata[column_name]['type'] = prediction['type']
            self.column_metadata[column_name]['confidence'] = prediction['confidence']
            self.column_metadata[column_name]['relationships'] = prediction.get('relationships', [])
            self.statistics['columns_classified'] += 1
            
            if self.streaming:
                await self.streaming.publish('column_classified', {
                    'column': column_name,
                    'type': prediction['type'],
                    'confidence': prediction['confidence']
                })
    
    async def _phase3_resolution(self):
        logger.info("Phase 3: Entity resolution and deduplication")
        
        resolved_entities = await self.entity_resolver.resolve(
            self.discovered_hosts,
            self.column_metadata
        )
        
        graph = self.graph_builder.build_entity_graph(resolved_entities)
        
        clusters = self.graph_builder.find_communities(graph)
        
        for cluster in clusters:
            master_entity = self.entity_resolver.select_master(cluster)
            
            for entity in cluster:
                if entity != master_entity:
                    self.discovered_hosts[master_entity].update(
                        self.discovered_hosts.pop(entity, {})
                    )
                    self.statistics['entities_resolved'] += 1
    
    async def _phase4_build(self):
        logger.info("Phase 4: Building CMDB")
        
        await self.cmdb.initialize()
        
        important_columns = self._rank_columns()
        
        await self.cmdb.create_schema(important_columns[:100])
        
        batch_size = 1000
        host_batch = []
        
        for hostname, data in self.discovered_hosts.items():
            enriched_data = await self.context_analyzer.enrich_host_data(
                hostname, data, self.column_metadata
            )
            
            host_batch.append(enriched_data)
            
            if len(host_batch) >= batch_size:
                await self.cmdb.insert_hosts(host_batch)
                host_batch = []
                
                self.statistics['hosts_discovered'] = len(self.discovered_hosts)
        
        if host_batch:
            await self.cmdb.insert_hosts(host_batch)
        
        await self.cmdb.create_indexes()
        await self.metadata_store.persist_metadata(self.column_metadata)
    
    async def _extract_features(self, batch):
        features = {}
        
        for table in batch:
            for column_name, values in table['columns'].items():
                if column_name not in features:
                    features[column_name] = await self.feature_extractor.extract(
                        column_name, 
                        values[:self.config['sampling']['column_sample_size']]
                    )
        
        return features
    
    async def _process_table(self, table_data, features):
        for row in table_data['rows']:
            potential_hosts = []
            
            for column_name, value in row.items():
                if value is None:
                    continue
                
                if column_name not in self.column_metadata:
                    self.column_metadata[column_name] = {
                        'samples': [],
                        'tables': set(),
                        'frequency': 0
                    }
                
                self.column_metadata[column_name]['samples'].append(value)
                self.column_metadata[column_name]['tables'].add(table_data['table_name'])
                self.column_metadata[column_name]['frequency'] += 1
                
                confidence = await self._is_hostname(column_name, value, features.get(column_name))
                
                if confidence > self.config['sampling']['confidence_threshold']:
                    potential_hosts.append({
                        'hostname': value,
                        'column': column_name,
                        'confidence': confidence,
                        'table': table_data['table_name']
                    })
            
            for host_info in potential_hosts:
                hostname = self._normalize_hostname(host_info['hostname'])
                
                if hostname not in self.discovered_hosts:
                    self.discovered_hosts[hostname] = {
                        'raw_forms': set(),
                        'occurrences': [],
                        'attributes': {}
                    }
                
                self.discovered_hosts[hostname]['raw_forms'].add(host_info['hostname'])
                self.discovered_hosts[hostname]['occurrences'].append({
                    'table': host_info['table'],
                    'column': host_info['column'],
                    'confidence': host_info['confidence']
                })
                
                for col, val in row.items():
                    if col != host_info['column'] and val:
                        if col not in self.discovered_hosts[hostname]['attributes']:
                            self.discovered_hosts[hostname]['attributes'][col] = []
                        self.discovered_hosts[hostname]['attributes'][col].append(val)
    
    async def _is_hostname(self, column_name, value, features):
        if features is None:
            return 0.0
        
        predictions = await self.ensemble.predict_single(column_name, value, features)
        
        hostname_types = ['hostname', 'fqdn', 'server', 'host', 'ip_address', 'instance_id']
        
        for pred in predictions:
            if pred['type'] in hostname_types:
                return pred['confidence']
        
        return 0.0
    
    def _normalize_hostname(self, hostname):
        import re
        hostname = str(hostname).lower().strip()
        hostname = re.sub(r'^https?://', '', hostname)
        hostname = re.sub(r':\d+$', '', hostname)
        hostname = hostname.split('/')[0]
        return hostname
    
    def _rank_columns(self):
        ranked = []
        
        for column_name, metadata in self.column_metadata.items():
            importance = metadata['frequency'] / max(sum(m['frequency'] for m in self.column_metadata.values()), 1)
            
            if metadata.get('type') in ['hostname', 'ip_address', 'owner', 'environment']:
                importance *= 2.0
            
            ranked.append({
                'name': column_name,
                'importance': importance,
                'type': metadata.get('type', 'unknown'),
                'confidence': metadata.get('confidence', 0)
            })
        
        ranked.sort(key=lambda x: x['importance'], reverse=True)
        return ranked
    
    def _report_statistics(self):
        elapsed = (datetime.now() - self.start_time).total_seconds()
        
        logger.info(f"""
        CMDB Discovery Complete:
        - Tables processed: {self.statistics['tables_processed']}
        - Rows scanned: {self.statistics['rows_scanned']:,}
        - Hosts discovered: {self.statistics['hosts_discovered']:,}
        - Columns classified: {self.statistics['columns_classified']}
        - Entities resolved: {self.statistics['entities_resolved']}
        - Time elapsed: {elapsed:.1f}s
        - Processing rate: {self.statistics['rows_scanned']/elapsed:.0f} rows/sec
        """)