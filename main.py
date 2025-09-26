#!/usr/bin/env python3
"""
Enhanced BigQuery CMDB Discovery System
Scans ALL BigQuery projects, datasets, and tables to build comprehensive CMDB
"""

import asyncio
import json
import logging
import argparse
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import torch

# Core components
from core.bigquery_scanner import BigQueryScanner
from core.cmdb_discovery import CMDBDiscovery
from core.cmdb_builder import CMDBBuilder
from core.classifier import IntelligentClassifier
from core.relationship_mapper import RelationshipMapper
from core.quality_analyzer import QualityAnalyzer
from core.config_manager import ConfigManager
from core.checkpoint_manager import CheckpointManager
from core.orchestrator import DiscoveryOrchestrator

# ML Models
from models.ensemble_predictor import EnsemblePredictor
from models.sherlock_model import SherlockModel
from models.sato_model import SatoModel
from models.doduo_model import DoduoModel
from models.llm_classifier import LLMClassifier

# Processors
from processors.feature_extractor import AdvancedFeatureExtractor
from processors.entity_resolver import EntityResolver
from processors.context_analyzer import ContextAnalyzer
from processors.graph_builder import GraphBuilder

# Storage
from storage.metadata_store import MetadataStore
from storage.streaming_handler import StreamingHandler

# Utils
from utils.logger import setup_logging, PerformanceLogger
from utils.gpu_optimizer import GPUOptimizer
from utils.data_loader import FastTabularDataLoader, CachedDataLoader

logger = setup_logging('bigquery_cmdb')

class EnhancedBigQueryCMDBSystem:
    """Enhanced system that uses ALL components to scan BigQuery comprehensively"""
    
    def __init__(self, config_path: str = 'config.json'):
        # Load and validate configuration
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.load_config()
        
        # Initialize GPU if available (Mac MPS)
        self.device = self._initialize_device()
        
        # Initialize core components
        self.scanner = BigQueryScanner(self.config)
        self.discovery = CMDBDiscovery(self.config)
        self.classifier = IntelligentClassifier()
        self.relationship_mapper = RelationshipMapper()
        self.quality_analyzer = QualityAnalyzer()
        
        # Initialize ML models
        self._initialize_ml_models()
        
        # Initialize processors
        self.feature_extractor = AdvancedFeatureExtractor(self.device)
        self.entity_resolver = EntityResolver(self.config, self.device)
        self.context_analyzer = ContextAnalyzer(self.device)
        self.graph_builder = GraphBuilder(self.config)
        
        # Initialize storage
        self.builder = CMDBBuilder(self.config.get('output_database', 'bigquery_cmdb.db'))
        self.metadata_store = MetadataStore(self.config)
        self.streaming_handler = StreamingHandler(self.config)
        
        # Initialize checkpoint manager
        self.checkpoint_manager = CheckpointManager()
        
        # Data loaders for efficient processing
        self.data_loader = FastTabularDataLoader(device=self.device)
        self.cache_loader = CachedDataLoader()
        
        # Performance tracking
        self.perf_logger = PerformanceLogger('performance')
        
        # Discovery orchestrator for complex workflows
        self.orchestrator = DiscoveryOrchestrator(self.config)
        
        # Statistics
        self.stats = {
            'start_time': datetime.now(),
            'projects_scanned': 0,
            'datasets_scanned': 0,
            'tables_scanned': 0,
            'rows_processed': 0,
            'hosts_discovered': 0,
            'unique_hosts': 0,
            'relationships_found': 0,
            'columns_classified': 0,
            'hostname_patterns_learned': 0,
            'scan_strategies_used': {}
        }
        
        # Learned patterns storage
        self.learned_hostname_patterns = []
        self.associated_columns = {}
    
    def _initialize_device(self) -> str:
        """Initialize GPU/MPS if available"""
        try:
            gpu_optimizer = GPUOptimizer()
            device = gpu_optimizer.initialize()
            logger.info(f"✅ GPU initialized: {device}")
            return device
        except Exception as e:
            logger.info(f"GPU not available, using CPU: {e}")
            return 'cpu'
    
    def _initialize_ml_models(self):
        """Initialize all ML models for column classification"""
        model_config = {
            'model_config': {
                'sherlock_features': 1588,
                'sato_topics': 400,
                'llm_model': 'rule-based',
                'llm_batch_size': 50,
                'weak_supervision_ratio': 0.013
            },
            'entity_resolution': {
                'comparison_threshold': 0.7,
                'confidence_levels': [0.8, 0.9, 0.95]
            }
        }
        
        self.ensemble_predictor = EnsemblePredictor(model_config, self.device)
        logger.info("✅ ML models initialized")
    
    async def run(self):
        """Main execution with comprehensive scanning"""
        logger.info("="*80)
        logger.info("🚀 ENHANCED BIGQUERY CMDB DISCOVERY SYSTEM")
        logger.info("="*80)
        
        # Check for checkpoint
        if self.checkpoint_manager.load():
            logger.info("📥 Resuming from checkpoint...")
            state = self.checkpoint_manager.load()
            if state:
                self.stats.update(state.get('statistics', {}))
                self.learned_hostname_patterns = state.get('patterns', [])
        
        try:
            # Step 1: Comprehensive BigQuery Scanning
            logger.info("\n📊 Step 1: Scanning ALL BigQuery projects, datasets, and tables...")
            bigquery_data = await self._comprehensive_scan()
            
            # Step 2: ML-based Column Classification
            logger.info("\n🤖 Step 2: Classifying columns using ML ensemble...")
            column_classifications = await self._classify_all_columns(bigquery_data)
            
            # Step 3: Pattern Learning
            logger.info("\n🧠 Step 3: Learning hostname patterns from discovered data...")
            await self._learn_hostname_patterns(bigquery_data, column_classifications)
            
            # Step 4: Comprehensive Host Discovery
            logger.info("\n🔍 Step 4: Discovering ALL hosts using learned patterns...")
            discovered_hosts = await self._discover_all_hosts(bigquery_data, column_classifications)
            
            # Step 5: Entity Resolution
            logger.info("\n🔗 Step 5: Resolving and deduplicating entities...")
            resolved_hosts = await self._resolve_entities(discovered_hosts)
            
            # Step 6: Context Enrichment
            logger.info("\n✨ Step 6: Enriching hosts with context...")
            enriched_hosts = await self._enrich_hosts(resolved_hosts, column_classifications)
            
            # Step 7: Relationship Discovery
            logger.info("\n🕸️ Step 7: Discovering relationships...")
            relationships = await self._discover_relationships(enriched_hosts)
            
            # Step 8: Quality Analysis
            logger.info("\n📈 Step 8: Analyzing data quality...")
            quality_report = await self._analyze_quality(enriched_hosts, relationships)
            
            # Step 9: Build Knowledge Graph
            logger.info("\n🌐 Step 9: Building knowledge graph...")
            graph = await self._build_graph(enriched_hosts, relationships)
            
            # Step 10: Build Final CMDB
            logger.info("\n🏗️ Step 10: Building comprehensive CMDB...")
            await self._build_final_cmdb(enriched_hosts, relationships, graph)
            
            # Step 11: Stream Results (if Kafka configured)
            logger.info("\n📡 Step 11: Streaming results...")
            await self._stream_results(enriched_hosts, relationships)
            
            # Step 12: Export Results
            logger.info("\n📁 Step 12: Exporting results...")
            await self._export_results(enriched_hosts, relationships, quality_report)
            
            # Save checkpoint
            self._save_checkpoint(enriched_hosts)
            
            # Print comprehensive summary
            self._print_comprehensive_summary(quality_report)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Discovery failed: {e}", exc_info=True)
            self._save_checkpoint(None)
            return False
    
    async def _comprehensive_scan(self) -> List[Dict]:
        """Scan ALL BigQuery projects, datasets, and tables"""
        all_data = []
        
        # Ensure no filtering
        self.scanner.datasets_filter = set()  # Empty = scan all
        self.scanner.tables_filter = set()    # Empty = scan all
        
        # Scan with progress tracking
        for project_id in self.config['bigquery']['projects']:
            logger.info(f"📂 Scanning project: {project_id}")
            
            try:
                # Get client for project
                client = self.scanner._get_client(project_id)
                
                # List ALL datasets
                datasets = list(client.list_datasets())
                logger.info(f"  Found {len(datasets)} datasets")
                
                for dataset_ref in datasets:
                    dataset_id = dataset_ref.dataset_id
                    logger.info(f"  📁 Scanning dataset: {dataset_id}")
                    
                    # List ALL tables in dataset
                    tables = list(client.list_tables(f"{project_id}.{dataset_id}"))
                    logger.info(f"    Found {len(tables)} tables")
                    
                    # Process EVERY table
                    for table_ref in tables:
                        table_id = table_ref.table_id
                        
                        # Use multi-strategy scanning
                        table_data = await self._scan_table_with_ml(
                            client, project_id, dataset_id, table_id
                        )
                        
                        if table_data:
                            all_data.append({
                                'type': 'bigquery',
                                'source': f'{project_id}.{dataset_id}.{table_id}',
                                'project': project_id,
                                'dataset': dataset_id,
                                'table': table_id,
                                'tables': [table_data]
                            })
                            self.stats['tables_scanned'] += 1
                    
                    self.stats['datasets_scanned'] += 1
                    
                    # Stream progress
                    await self.streaming_handler.publish('scan_progress', {
                        'project': project_id,
                        'dataset': dataset_id,
                        'tables_scanned': self.stats['tables_scanned']
                    })
                
                self.stats['projects_scanned'] += 1
                
            except Exception as e:
                logger.error(f"Error scanning project {project_id}: {e}")
        
        logger.info(f"✅ Scanned {self.stats['tables_scanned']} tables total")
        return all_data
    
    async def _scan_table_with_ml(self, client, project_id: str, dataset_id: str, table_id: str) -> Dict:
        """Scan table with ML-enhanced column detection using multiple strategies"""
        full_table_id = f"{project_id}.{dataset_id}.{table_id}"
        
        # Define multiple scanning strategies
        strategies = [
            ('iterator_simple', self._scan_strategy_iterator_simple),
            ('iterator_with_conversion', self._scan_strategy_iterator_with_conversion),
            ('to_arrow', self._scan_strategy_arrow),
            ('to_dataframe_safe', self._scan_strategy_dataframe_safe),
            ('direct_sql', self._scan_strategy_direct_sql),
            ('fallback_minimal', self._scan_strategy_fallback_minimal)
        ]
        
        # Try each strategy until one works
        for strategy_name, strategy_func in strategies:
            try:
                logger.debug(f"Attempting strategy: {strategy_name} for table {table_id}")
                result = await strategy_func(client, project_id, dataset_id, table_id)
                
                if result is not None:
                    logger.info(f"✅ Successfully scanned {table_id} using strategy: {strategy_name}")
                    
                    # Track which strategy worked
                    if strategy_name not in self.stats['scan_strategies_used']:
                        self.stats['scan_strategies_used'][strategy_name] = 0
                    self.stats['scan_strategies_used'][strategy_name] += 1
                    
                    return result
                    
            except Exception as e:
                logger.debug(f"Strategy {strategy_name} failed for {table_id}: {str(e)[:100]}")
                continue
        
        # If all strategies fail, return minimal data
        logger.warning(f"All strategies failed for {full_table_id}, returning minimal data")
        return await self._get_minimal_table_info(client, project_id, dataset_id, table_id)
    
    async def _scan_strategy_iterator_simple(self, client, project_id: str, dataset_id: str, table_id: str) -> Dict:
        """Strategy 1: Simple iterator without pandas"""
        full_table_id = f"{project_id}.{dataset_id}.{table_id}"
        
        table = client.get_table(full_table_id)
        if table.num_rows == 0:
            return None
        
        sample_size = min(
            int(table.num_rows * self.config['bigquery']['sample_percent'] / 100),
            self.config['bigquery'].get('max_rows_per_table', 100000) or 100000
        )
        
        query = f"""
        SELECT *
        FROM `{full_table_id}`
        TABLESAMPLE SYSTEM ({min(self.config['bigquery']['sample_percent'], 100)} PERCENT)
        LIMIT {sample_size}
        """
        
        query_job = client.query(query)
        rows = []
        columns_with_features = {}
        
        # Initialize columns from schema
        for field in table.schema:
            columns_with_features[field.name] = {
                'name': field.name,
                'samples': [],
                'features': None,
                'statistics': {},
                'type': field.field_type,
                'mode': field.mode
            }
        
        # Process rows without pandas
        for row in query_job.result():
            row_dict = {}
            for field in table.schema:
                value = row.get(field.name)
                
                # Handle special types
                if value is not None:
                    if field.field_type in ['RECORD', 'STRUCT', 'JSON']:
                        value = str(value)
                    elif field.mode == 'REPEATED':
                        value = str(value) if value else '[]'
                    elif hasattr(value, 'isoformat'):
                        value = value.isoformat()
                
                row_dict[field.name] = value
                
                # Collect samples
                if len(columns_with_features[field.name]['samples']) < 100:
                    columns_with_features[field.name]['samples'].append(value)
            
            rows.append(row_dict)
        
        # Extract features for each column
        for col_name, col_data in columns_with_features.items():
            values = col_data['samples']
            
            # Extract features using feature extractor
            features = await self.feature_extractor.extract(col_name, values)
            col_data['features'] = features
            col_data['statistics'] = self._analyze_column(col_name, values)
        
        self.stats['rows_processed'] += len(rows)
        
        return {
            'name': table_id,
            'full_name': full_table_id,
            'rows': rows,
            'columns': columns_with_features,
            'row_count': len(rows),
            'total_rows': table.num_rows
        }
    
    async def _scan_strategy_iterator_with_conversion(self, client, project_id: str, dataset_id: str, table_id: str) -> Dict:
        """Strategy 2: Iterator with TO_JSON_STRING for complex types"""
        full_table_id = f"{project_id}.{dataset_id}.{table_id}"
        
        table = client.get_table(full_table_id)
        if table.num_rows == 0:
            return None
        
        # Build query with JSON conversion for complex types
        select_parts = []
        complex_fields = []
        
        for field in table.schema:
            if field.mode == 'REPEATED' or field.field_type in ['RECORD', 'STRUCT', 'JSON', 'GEOGRAPHY', 'ARRAY']:
                select_parts.append(f"TO_JSON_STRING({field.name}) AS {field.name}")
                complex_fields.append(field.name)
            else:
                select_parts.append(field.name)
        
        sample_size = min(
            int(table.num_rows * self.config['bigquery']['sample_percent'] / 100),
            self.config['bigquery'].get('max_rows_per_table', 100000) or 100000
        )
        
        query = f"""
        SELECT {', '.join(select_parts)}
        FROM `{full_table_id}`
        TABLESAMPLE SYSTEM ({min(self.config['bigquery']['sample_percent'], 100)} PERCENT)
        LIMIT {sample_size}
        """
        
        query_job = client.query(query)
        rows = []
        columns_with_features = {}
        
        # Process results
        for row in query_job.result():
            row_dict = dict(row)
            rows.append(row_dict)
            
            for key, value in row_dict.items():
                if key not in columns_with_features:
                    columns_with_features[key] = {
                        'name': key,
                        'samples': [],
                        'features': None,
                        'statistics': {}
                    }
                
                if len(columns_with_features[key]['samples']) < 100:
                    columns_with_features[key]['samples'].append(value)
        
        # Extract features
        for col_name, col_data in columns_with_features.items():
            values = col_data['samples']
            features = await self.feature_extractor.extract(col_name, values)
            col_data['features'] = features
            col_data['statistics'] = self._analyze_column(col_name, values)
        
        self.stats['rows_processed'] += len(rows)
        
        return {
            'name': table_id,
            'full_name': full_table_id,
            'rows': rows,
            'columns': columns_with_features,
            'row_count': len(rows),
            'total_rows': table.num_rows
        }
    
    async def _scan_strategy_arrow(self, client, project_id: str, dataset_id: str, table_id: str) -> Dict:
        """Strategy 3: Use PyArrow for better type handling"""
        full_table_id = f"{project_id}.{dataset_id}.{table_id}"
        
        table = client.get_table(full_table_id)
        if table.num_rows == 0:
            return None
        
        sample_size = min(
            int(table.num_rows * self.config['bigquery']['sample_percent'] / 100),
            self.config['bigquery'].get('max_rows_per_table', 100000) or 100000
        )
        
        query = f"""
        SELECT *
        FROM `{full_table_id}`
        TABLESAMPLE SYSTEM ({min(self.config['bigquery']['sample_percent'], 100)} PERCENT)
        LIMIT {sample_size}
        """
        
        query_job = client.query(query)
        
        # Try to use Arrow
        arrow_table = query_job.to_arrow()
        
        rows = []
        columns_with_features = {}
        
        # Convert Arrow table to Python objects
        for col_name in arrow_table.column_names:
            column = arrow_table.column(col_name)
            
            columns_with_features[col_name] = {
                'name': col_name,
                'samples': [],
                'features': None,
                'statistics': {}
            }
            
            # Convert to Python list safely
            for i in range(min(len(column), sample_size)):
                value = column[i].as_py() if hasattr(column[i], 'as_py') else str(column[i])
                
                if i < 100:
                    columns_with_features[col_name]['samples'].append(value)
                
                if i < len(rows):
                    rows[i][col_name] = value
                else:
                    rows.append({col_name: value})
        
        # Extract features
        for col_name, col_data in columns_with_features.items():
            values = col_data['samples']
            features = await self.feature_extractor.extract(col_name, values)
            col_data['features'] = features
            col_data['statistics'] = self._analyze_column(col_name, values)
        
        self.stats['rows_processed'] += len(rows)
        
        return {
            'name': table_id,
            'full_name': full_table_id,
            'rows': rows,
            'columns': columns_with_features,
            'row_count': len(rows),
            'total_rows': table.num_rows
        }
    
    async def _scan_strategy_dataframe_safe(self, client, project_id: str, dataset_id: str, table_id: str) -> Dict:
        """Strategy 4: Use DataFrame with safe conversion"""
        full_table_id = f"{project_id}.{dataset_id}.{table_id}"
        
        table = client.get_table(full_table_id)
        if table.num_rows == 0:
            return None
        
        sample_size = min(
            int(table.num_rows * self.config['bigquery']['sample_percent'] / 100),
            self.config['bigquery'].get('max_rows_per_table', 100000) or 100000
        )
        
        query = f"""
        SELECT *
        FROM `{full_table_id}`
        TABLESAMPLE SYSTEM ({min(self.config['bigquery']['sample_percent'], 100)} PERCENT)
        LIMIT {sample_size}
        """
        
        query_job = client.query(query)
        
        # Use to_dataframe with specific dtypes to avoid issues
        import pandas as pd
        import numpy as np
        
        # Set string dtype for problematic columns
        dtype_kwargs = {}
        for field in table.schema:
            if field.field_type in ['RECORD', 'STRUCT', 'JSON', 'ARRAY'] or field.mode == 'REPEATED':
                dtype_kwargs[field.name] = 'object'
        
        if dtype_kwargs:
            df = query_job.to_dataframe(create_bqstorage_client=False, dtypes=dtype_kwargs)
        else:
            df = query_job.to_dataframe(create_bqstorage_client=False)
        
        # Convert DataFrame to safe format
        rows = []
        columns_with_features = {}
        
        for col_name in df.columns:
            columns_with_features[col_name] = {
                'name': col_name,
                'samples': [],
                'features': None,
                'statistics': {}
            }
            
            # Get samples safely
            col_values = df[col_name]
            for i in range(min(100, len(col_values))):
                try:
                    value = col_values.iloc[i]
                    
                    # Handle pandas NA and numpy arrays
                    if pd.isna(value):
                        value = None
                    elif isinstance(value, np.ndarray):
                        if value.size > 0:
                            value = value.tolist()
                        else:
                            value = []
                    elif hasattr(value, 'tolist'):
                        value = value.tolist()
                    elif hasattr(value, 'item'):
                        value = value.item()
                    else:
                        value = str(value) if value is not None else None
                    
                    columns_with_features[col_name]['samples'].append(value)
                except Exception as e:
                    logger.debug(f"Error processing value in {col_name}: {e}")
                    columns_with_features[col_name]['samples'].append(None)
        
        # Convert rows
        for idx in range(len(df)):
            row_dict = {}
            for col in df.columns:
                try:
                    value = df.iloc[idx][col]
                    
                    # Safe conversion
                    if pd.isna(value):
                        value = None
                    elif isinstance(value, np.ndarray):
                        if value.size > 0:
                            value = value.tolist()
                        else:
                            value = []
                    elif hasattr(value, 'tolist'):
                        value = value.tolist()
                    elif hasattr(value, 'item'):
                        value = value.item()
                    else:
                        value = str(value) if value is not None else None
                    
                    row_dict[col] = value
                except Exception as e:
                    logger.debug(f"Error converting row value: {e}")
                    row_dict[col] = None
            
            rows.append(row_dict)
        
        # Extract features
        for col_name, col_data in columns_with_features.items():
            values = col_data['samples']
            features = await self.feature_extractor.extract(col_name, values)
            col_data['features'] = features
            col_data['statistics'] = self._analyze_column(col_name, values)
        
        self.stats['rows_processed'] += len(rows)
        
        return {
            'name': table_id,
            'full_name': full_table_id,
            'rows': rows,
            'columns': columns_with_features,
            'row_count': len(rows),
            'total_rows': table.num_rows
        }
    
    async def _scan_strategy_direct_sql(self, client, project_id: str, dataset_id: str, table_id: str) -> Dict:
        """Strategy 5: Direct SQL with explicit casting"""
        full_table_id = f"{project_id}.{dataset_id}.{table_id}"
        
        table = client.get_table(full_table_id)
        if table.num_rows == 0:
            return None
        
        # Get column info first
        info_query = f"""
        SELECT 
            column_name,
            data_type
        FROM `{project_id}.{dataset_id}.INFORMATION_SCHEMA.COLUMNS`
        WHERE table_name = '{table_id}'
        """
        
        try:
            info_job = client.query(info_query)
            column_info = {row['column_name']: row['data_type'] for row in info_job.result()}
        except:
            # Fallback to schema
            column_info = {field.name: field.field_type for field in table.schema}
        
        # Build safe query with casting
        select_parts = []
        for col_name, col_type in column_info.items():
            if col_type in ['ARRAY', 'STRUCT', 'JSON', 'GEOGRAPHY']:
                select_parts.append(f"CAST({col_name} AS STRING) AS {col_name}")
            else:
                select_parts.append(col_name)
        
        if not select_parts:
            select_parts = ['*']
        
        sample_size = min(1000, table.num_rows)  # Smaller sample for safety
        
        query = f"""
        SELECT {', '.join(select_parts)}
        FROM `{full_table_id}`
        LIMIT {sample_size}
        """
        
        query_job = client.query(query)
        
        rows = []
        columns_with_features = {}
        
        for row in query_job.result():
            row_dict = dict(row)
            rows.append(row_dict)
            
            for key, value in row_dict.items():
                if key not in columns_with_features:
                    columns_with_features[key] = {
                        'name': key,
                        'samples': [],
                        'features': None,
                        'statistics': {}
                    }
                
                if len(columns_with_features[key]['samples']) < 100:
                    columns_with_features[key]['samples'].append(value)
        
        # Extract features
        for col_name, col_data in columns_with_features.items():
            values = col_data['samples']
            features = await self.feature_extractor.extract(col_name, values)
            col_data['features'] = features
            col_data['statistics'] = self._analyze_column(col_name, values)
        
        self.stats['rows_processed'] += len(rows)
        
        return {
            'name': table_id,
            'full_name': full_table_id,
            'rows': rows,
            'columns': columns_with_features,
            'row_count': len(rows),
            'total_rows': table.num_rows
        }
    
    async def _scan_strategy_fallback_minimal(self, client, project_id: str, dataset_id: str, table_id: str) -> Dict:
        """Strategy 6: Minimal scan - just get schema and a few rows"""
        full_table_id = f"{project_id}.{dataset_id}.{table_id}"
        
        table = client.get_table(full_table_id)
        
        columns_with_features = {}
        rows = []
        
        # Just get schema info
        for field in table.schema:
            columns_with_features[field.name] = {
                'name': field.name,
                'type': field.field_type,
                'mode': field.mode,
                'samples': [],
                'features': None,
                'statistics': {}
            }
        
        # Try to get just a few rows
        try:
            query = f"SELECT * FROM `{full_table_id}` LIMIT 10"
            query_job = client.query(query)
            
            for row in query_job.result():
                row_dict = {}
                for field in table.schema:
                    try:
                        value = row.get(field.name)
                        row_dict[field.name] = str(value) if value is not None else None
                        
                        if len(columns_with_features[field.name]['samples']) < 10:
                            columns_with_features[field.name]['samples'].append(str(value) if value is not None else None)
                    except:
                        row_dict[field.name] = None
                
                rows.append(row_dict)
        except:
            logger.warning(f"Could not sample rows from {table_id}")
        
        self.stats['rows_processed'] += len(rows)
        
        return {
            'name': table_id,
            'full_name': full_table_id,
            'rows': rows,
            'columns': columns_with_features,
            'row_count': len(rows),
            'total_rows': table.num_rows,
            'scan_type': 'minimal'
        }
    
    async def _get_minimal_table_info(self, client, project_id: str, dataset_id: str, table_id: str) -> Dict:
        """Get absolute minimal info when all strategies fail"""
        full_table_id = f"{project_id}.{dataset_id}.{table_id}"
        
        try:
            table = client.get_table(full_table_id)
            
            return {
                'name': table_id,
                'full_name': full_table_id,
                'rows': [],
                'columns': {field.name: {'name': field.name, 'type': field.field_type} for field in table.schema},
                'row_count': 0,
                'total_rows': table.num_rows,
                'scan_type': 'metadata_only',
                'error': 'All scan strategies failed'
            }
        except:
            return {
                'name': table_id,
                'full_name': full_table_id,
                'rows': [],
                'columns': {},
                'row_count': 0,
                'total_rows': 0,
                'scan_type': 'failed',
                'error': 'Could not access table'
            }
    
    async def _classify_all_columns(self, bigquery_data: List[Dict]) -> Dict:
        """Classify ALL columns using ensemble ML models"""
        all_classifications = {}
        
        for data_source in bigquery_data:
            for table_data in data_source.get('tables', []):
                columns = table_data.get('columns', {})
                
                for col_name, col_info in columns.items():
                    # Use ensemble predictor for classification
                    classification = await self.ensemble_predictor.classify_columns(
                        {col_name: col_info},
                        {}
                    )
                    
                    all_classifications[f"{data_source['source']}:{col_name}"] = classification.get(col_name, {})
                    self.stats['columns_classified'] += 1
        
        # Persist classifications
        await self.metadata_store.persist_metadata(all_classifications)
        
        logger.info(f"✅ Classified {self.stats['columns_classified']} columns")
        return all_classifications
    
    async def _learn_hostname_patterns(self, bigquery_data: List[Dict], classifications: Dict):
        """Learn hostname patterns from discovered data"""
        hostname_samples = []
        
        # Collect all identified hostname columns
        for key, classification in classifications.items():
            if classification.get('type') in ['hostname', 'server', 'host', 'instance']:
                # Extract table and column from key
                parts = key.split(':')
                if len(parts) == 2:
                    table_source = parts[0]
                    col_name = parts[1]
                    
                    # Find the actual data
                    for data_source in bigquery_data:
                        if data_source.get('source') == table_source:
                            for table_data in data_source.get('tables', []):
                                if col_name in table_data.get('columns', {}):
                                    samples = table_data['columns'][col_name].get('samples', [])
                                    hostname_samples.extend(samples[:50])
        
        if hostname_samples:
            # Learn patterns using ML
            patterns = self._extract_patterns(hostname_samples)
            self.learned_hostname_patterns.extend(patterns)
            self.stats['hostname_patterns_learned'] = len(patterns)
            
            logger.info(f"✅ Learned {len(patterns)} hostname patterns")
            
            # Find associated columns
            await self._find_associated_columns(bigquery_data, classifications)
    
    def _extract_patterns(self, samples: List) -> List[Dict]:
        """Extract patterns from hostname samples"""
        patterns = []
        
        # Analyze naming conventions
        for sample in samples[:100]:
            if sample:
                sample_str = str(sample).lower()
                
                # Check for common patterns
                if '-' in sample_str:
                    parts = sample_str.split('-')
                    pattern = {
                        'type': 'hyphenated',
                        'segments': len(parts),
                        'example': sample_str,
                        'regex': r'^[a-z0-9]+(-[a-z0-9]+)*$'
                    }
                    patterns.append(pattern)
                
                elif '.' in sample_str:
                    parts = sample_str.split('.')
                    pattern = {
                        'type': 'fqdn',
                        'segments': len(parts),
                        'example': sample_str,
                        'regex': r'^[a-z0-9]+(\.[a-z0-9]+)*$'
                    }
                    patterns.append(pattern)
        
        # Deduplicate patterns
        unique_patterns = []
        seen = set()
        for pattern in patterns:
            key = f"{pattern['type']}:{pattern['segments']}"
            if key not in seen:
                unique_patterns.append(pattern)
                seen.add(key)
        
        return unique_patterns
    
    async def _find_associated_columns(self, bigquery_data: List[Dict], classifications: Dict):
        """Find columns that frequently appear with hostname columns"""
        from collections import Counter
        
        cooccurrence = {}
        
        for data_source in bigquery_data:
            for table_data in data_source.get('tables', []):
                columns = list(table_data.get('columns', {}).keys())
                
                # Find hostname columns in this table
                hostname_cols = []
                for col in columns:
                    key = f"{data_source['source']}:{col}"
                    if key in classifications:
                        if classifications[key].get('type') in ['hostname', 'server', 'host']:
                            hostname_cols.append(col)
                
                # Track co-occurrence
                if hostname_cols:
                    for hostname_col in hostname_cols:
                        if hostname_col not in cooccurrence:
                            cooccurrence[hostname_col] = {}
                        
                        for other_col in columns:
                            if other_col != hostname_col:
                                if other_col not in cooccurrence[hostname_col]:
                                    cooccurrence[hostname_col][other_col] = 0
                                cooccurrence[hostname_col][other_col] += 1
        
        # Find most common associations
        for hostname_col, associations in cooccurrence.items():
            if associations:
                # Sort by frequency
                sorted_assoc = sorted(associations.items(), key=lambda x: x[1], reverse=True)
                self.associated_columns[hostname_col] = [col for col, _ in sorted_assoc[:10]]
        
        logger.info(f"✅ Found associated columns for {len(self.associated_columns)} hostname columns")
    
    async def _discover_all_hosts(self, bigquery_data: List[Dict], classifications: Dict) -> Dict:
        """Discover ALL unique hosts from ALL tables"""
        all_discovered_hosts = {}
        
        # Process each table
        for data_source in bigquery_data:
            # Use discovery module with learned patterns
            hosts = await self.discovery.discover_hosts([data_source])
            
            # Merge with existing hosts
            for hostname, host_data in hosts.items():
                if hostname not in all_discovered_hosts:
                    all_discovered_hosts[hostname] = host_data
                else:
                    # Merge data from multiple sources
                    existing = all_discovered_hosts[hostname]
                    existing['sources'].extend(host_data.get('sources', []))
                    
                    # Merge attributes
                    for attr, values in host_data.get('attributes', {}).items():
                        if attr not in existing['attributes']:
                            existing['attributes'][attr] = []
                        if isinstance(values, list):
                            existing['attributes'][attr].extend(values)
                        else:
                            existing['attributes'][attr].append(values)
        
        self.stats['hosts_discovered'] = len(all_discovered_hosts)
        
        # Stream discoveries
        await self.streaming_handler.stream_discoveries(all_discovered_hosts)
        
        logger.info(f"✅ Discovered {len(all_discovered_hosts)} total unique hosts")
        return all_discovered_hosts
    
    async def _resolve_entities(self, discovered_hosts: Dict) -> Dict:
        """Resolve and deduplicate entities"""
        resolved = await self.entity_resolver.resolve(
            discovered_hosts,
            self.associated_columns
        )
        
        self.stats['unique_hosts'] = len(resolved)
        logger.info(f"✅ Resolved to {len(resolved)} unique hosts")
        return resolved
    
    async def _enrich_hosts(self, resolved_hosts: Dict, classifications: Dict) -> List[Dict]:
        """Enrich hosts with context and classification"""
        enriched = []
        
        for hostname, host_data in resolved_hosts.items():
            # Classify the host
            classification = await self.classifier.classify_host(hostname, host_data)
            
            # Enrich with context
            enriched_host = await self.context_analyzer.enrich_host_data(
                hostname,
                host_data,
                classifications
            )
            
            enriched_host['classification'] = classification
            enriched.append(enriched_host)
        
        # Index in metadata store
        await self.metadata_store.index_hosts(enriched)
        
        logger.info(f"✅ Enriched {len(enriched)} hosts")
        return enriched
    
    async def _discover_relationships(self, enriched_hosts: List[Dict]) -> List[Dict]:
        """Discover relationships between hosts"""
        # Convert list to dict for relationship mapper
        hosts_dict = {h['hostname']: h for h in enriched_hosts}
        
        relationships = self.relationship_mapper.map_relationships(hosts_dict)
        
        self.stats['relationships_found'] = len(relationships)
        
        # Stream relationships
        await self.streaming_handler.stream_relationships(relationships)
        
        # Store in metadata store
        await self.metadata_store.create_graph_relationships(relationships)
        
        logger.info(f"✅ Found {len(relationships)} relationships")
        return relationships
    
    async def _analyze_quality(self, hosts: List[Dict], relationships: List[Dict]) -> Dict:
        """Analyze data quality"""
        cmdb_data = {
            'entities': hosts,
            'relationships': relationships
        }
        
        quality_report = self.quality_analyzer.analyze(cmdb_data)
        anomalies = self.quality_analyzer.detect_anomalies(cmdb_data)
        insights = self.quality_analyzer.generate_insights(cmdb_data)
        
        quality_report['anomalies'] = anomalies
        quality_report['insights'] = insights
        
        logger.info(f"✅ Quality score: {quality_report['overall_quality_score']:.2%}")
        return quality_report
    
    async def _build_graph(self, hosts: List[Dict], relationships: List[Dict]) -> Dict:
        """Build knowledge graph"""
        graph = await self.graph_builder.build(hosts, relationships)
        
        logger.info(f"✅ Built graph with {len(graph.get('nodes', []))} nodes")
        return graph
    
    async def _build_final_cmdb(self, hosts: List[Dict], relationships: List[Dict], graph: Dict):
        """Build the final CMDB database"""
        await self.builder.initialize()
        
        # Create schema with all discovered columns
        all_columns = set()
        for host in hosts:
            all_columns.update(host.keys())
        
        columns = [{'name': col, 'type': 'string'} for col in all_columns]
        await self.builder.create_schema(columns)
        
        # Insert hosts
        await self.builder.insert_hosts(hosts)
        
        # Insert relationships
        await self.builder.insert_relationships(relationships)
        
        # Create indexes
        await self.builder.create_indexes()
        
        # Get statistics
        stats = self.builder.get_statistics()
        logger.info(f"✅ CMDB built with {stats.get('total_hosts', 0)} hosts")
    
    async def _stream_results(self, hosts: List[Dict], relationships: List[Dict]):
        """Stream results to Kafka if configured"""
        if self.streaming_handler.producer:
            # Stream hosts in batches
            for i in range(0, len(hosts), 100):
                batch = hosts[i:i+100]
                await self.streaming_handler.publish_batch('cmdb_hosts', batch)
            
            # Stream relationships
            for i in range(0, len(relationships), 100):
                batch = relationships[i:i+100]
                await self.streaming_handler.publish_batch('cmdb_relationships', batch)
            
            logger.info("✅ Streamed results to Kafka")
    
    async def _export_results(self, hosts: List[Dict], relationships: List[Dict], quality_report: Dict):
        """Export results in multiple formats"""
        output_dir = Path(self.config.get('export', {}).get('output_dir', 'output'))
        output_dir.mkdir(exist_ok=True)
        
        # Export hosts
        import csv
        csv_file = output_dir / f"cmdb_hosts_{datetime.now():%Y%m%d_%H%M%S}.csv"
        with open(csv_file, 'w', newline='') as f:
            if hosts:
                writer = csv.DictWriter(f, fieldnames=hosts[0].keys())
                writer.writeheader()
                writer.writerows(hosts)
        
        # Export as JSON
        json_file = output_dir / f"cmdb_complete_{datetime.now():%Y%m%d_%H%M%S}.json"
        with open(json_file, 'w') as f:
            json.dump({
                'hosts': hosts,
                'relationships': relationships,
                'quality_report': quality_report,
                'statistics': self.stats,
                'learned_patterns': self.learned_hostname_patterns,
                'associated_columns': self.associated_columns
            }, f, indent=2, default=str)
        
        # Export to Parquet
        self.builder.export_to_parquet(str(output_dir / 'cmdb_hosts.parquet'))
        
        logger.info(f"✅ Exported results to {output_dir}")
    
    def _save_checkpoint(self, hosts):
        """Save checkpoint for resume capability"""
        state = {
            'statistics': self.stats,
            'patterns': self.learned_hostname_patterns,
            'associated_columns': self.associated_columns,
            'hosts': len(hosts) if hosts else 0
        }
        self.checkpoint_manager.save(state)
    
    def _analyze_column(self, column_name: str, values: List) -> Dict:
        """Analyze column statistics"""
        return {
            'unique_count': len(set(str(v) for v in values if v)),
            'null_count': sum(1 for v in values if v is None or v == ''),
            'total_count': len(values),
            'sample_values': list(set(str(v) for v in values[:10] if v))
        }
    
    def _print_comprehensive_summary(self, quality_report: Dict):
        """Print comprehensive discovery summary"""
        duration = (datetime.now() - self.stats['start_time']).total_seconds()
        
        print("\n" + "="*80)
        print("COMPREHENSIVE BIGQUERY CMDB DISCOVERY SUMMARY")
        print("="*80)
        print(f"Duration:                    {duration:.1f} seconds")
        print(f"Projects Scanned:            {self.stats['projects_scanned']}")
        print(f"Datasets Scanned:            {self.stats['datasets_scanned']}")
        print(f"Tables Scanned:              {self.stats['tables_scanned']}")
        print(f"Rows Processed:              {self.stats['rows_processed']:,}")
        print(f"Columns Classified:          {self.stats['columns_classified']}")
        print("-"*80)
        print(f"Hosts Discovered:            {self.stats['hosts_discovered']}")
        print(f"Unique Hosts (Resolved):     {self.stats['unique_hosts']}")
        print(f"Relationships Found:         {self.stats['relationships_found']}")
        print(f"Hostname Patterns Learned:   {self.stats['hostname_patterns_learned']}")
        print(f"Associated Columns Found:    {len(self.associated_columns)}")
        print("-"*80)
        print("Scan Strategies Used:")
        for strategy, count in self.stats.get('scan_strategies_used', {}).items():
            print(f"  {strategy:25s} {count:5d} tables")
        print("-"*80)
        print(f"Overall Quality Score:       {quality_report.get('overall_quality_score', 0):.2%}")
        print(f"Data Anomalies:              {len(quality_report.get('anomalies', []))}")
        print(f"Insights Generated:          {len(quality_report.get('insights', []))}")
        print("-"*80)
        print(f"CMDB Database:               {self.config['output_database']}")
        print(f"Device Used:                 {self.device}")
        print("="*80)
        
        # Print top insights
        if quality_report.get('insights'):
            print("\nTop Insights:")
            for i, insight in enumerate(quality_report['insights'][:5], 1):
                print(f"  {i}. {insight}")
        
        # Print learned patterns
        if self.learned_hostname_patterns:
            print("\nLearned Hostname Patterns:")
            for pattern in self.learned_hostname_patterns[:5]:
                print(f"  - {pattern['type']}: {pattern.get('example', 'N/A')}")
        
        print("\n✅ Discovery complete! Query your CMDB with:")
        print(f"   sqlite3 {self.config['output_database']}")
        print("="*80)

async def main():
    parser = argparse.ArgumentParser(
        description='Enhanced BigQuery CMDB Discovery - Comprehensive Infrastructure Scanner'
    )
    parser.add_argument('--config', default='config.json', help='Configuration file')
    parser.add_argument('--projects', nargs='+', help='Override BigQuery projects')
    parser.add_argument('--no-filter', action='store_true', help='Scan ALL datasets and tables')
    parser.add_argument('--learn-patterns', action='store_true', default=True, help='Learn hostname patterns')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    system = EnhancedBigQueryCMDBSystem(args.config)
    
    # Override projects if specified
    if args.projects:
        system.config['bigquery']['projects'] = args.projects
    
    # Ensure no filtering
    if args.no_filter:
        system.config['bigquery']['datasets_filter'] = []
        system.config['bigquery']['tables_filter'] = []
    
    success = await system.run()
    
    if success:
        logger.info("✅ Enhanced BigQuery CMDB Discovery completed successfully!")
    else:
        logger.error("❌ Discovery failed")
        sys.exit(1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)