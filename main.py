#!/usr/bin/env python3
"""
Smart CMDB Discovery System
Automatically discovers and classifies IT infrastructure from various data sources
"""

import asyncio
import json
import logging
import argparse
import signal
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

from core.orchestrator import DiscoveryOrchestrator
from core.config_manager import ConfigManager
from utils.logger import setup_logging

# Setup logging
logger = setup_logging('cmdb_discovery')

class CMDBDiscoverySystem:
    """Main CMDB Discovery System"""
    
    def __init__(self, config_path: str = 'config.json'):
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.load_config()
        self.orchestrator = None
        self.shutdown_requested = False
        
        # Register signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info("Shutdown signal received, saving state...")
        self.shutdown_requested = True
        if self.orchestrator:
            self.orchestrator.save_checkpoint()
        sys.exit(0)
    
    async def run(self, mode: str = 'discover'):
        """Run the discovery system"""
        logger.info("="*60)
        logger.info("Smart CMDB Discovery System v2.0")
        logger.info("="*60)
        
        # Validate environment
        if not self._validate_environment():
            logger.error("Environment validation failed")
            return False
        
        # Initialize orchestrator
        self.orchestrator = DiscoveryOrchestrator(self.config)
        
        # Load checkpoint if resuming
        if self.config.get('resume_enabled', True):
            self.orchestrator.load_checkpoint()
        
        try:
            if mode == 'discover':
                await self._run_discovery()
            elif mode == 'analyze':
                await self._run_analysis()
            elif mode == 'demo':
                await self._run_demo()
            else:
                logger.error(f"Unknown mode: {mode}")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"System error: {e}", exc_info=True)
            self.orchestrator.save_checkpoint()
            return False
    
    async def _run_discovery(self):
        """Run full discovery process"""
        logger.info("Starting discovery process...")
        
        # Phase 1: Data source discovery
        logger.info("Phase 1: Scanning data sources")
        discovered_data = await self.orchestrator.scan_data_sources()
        logger.info(f"Discovered {len(discovered_data)} data sources")
        
        # Phase 2: Host extraction
        logger.info("Phase 2: Extracting hosts")
        hosts = await self.orchestrator.extract_hosts(discovered_data)
        logger.info(f"Extracted {len(hosts)} potential hosts")
        
        # Phase 3: Classification
        logger.info("Phase 3: Classifying entities")
        classified = await self.orchestrator.classify_entities(hosts)
        logger.info(f"Classified {len(classified)} entities")
        
        # Phase 4: Relationship mapping
        logger.info("Phase 4: Mapping relationships")
        relationships = await self.orchestrator.map_relationships(classified)
        logger.info(f"Mapped {len(relationships)} relationships")
        
        # Phase 5: Build CMDB
        logger.info("Phase 5: Building CMDB")
        await self.orchestrator.build_cmdb(classified, relationships)
        
        # Generate report
        self._generate_report()
    
    async def _run_analysis(self):
        """Run analysis on existing CMDB"""
        logger.info("Running CMDB analysis...")
        
        # Load existing CMDB
        cmdb_data = await self.orchestrator.load_cmdb()
        
        if not cmdb_data:
            logger.error("No CMDB data found. Run discovery first.")
            return
        
        # Analyze data quality
        quality_report = await self.orchestrator.analyze_data_quality(cmdb_data)
        
        # Find anomalies
        anomalies = await self.orchestrator.detect_anomalies(cmdb_data)
        
        # Generate insights
        insights = await self.orchestrator.generate_insights(cmdb_data)
        
        # Print results
        self._print_analysis_results(quality_report, anomalies, insights)
    
    async def _run_demo(self):
        """Run demo with synthetic data"""
        logger.info("Running demo mode with synthetic data...")
        
        from utils.demo_generator import DemoDataGenerator
        
        generator = DemoDataGenerator()
        demo_data = generator.generate_demo_environment(
            n_hosts=self.config.get('demo_hosts', 100),
            n_applications=self.config.get('demo_apps', 20)
        )
        
        # Run discovery on demo data
        hosts = await self.orchestrator.extract_hosts(demo_data)
        classified = await self.orchestrator.classify_entities(hosts)
        relationships = await self.orchestrator.map_relationships(classified)
        await self.orchestrator.build_cmdb(classified, relationships)
        
        logger.info("Demo completed successfully!")
        self._generate_report()
    
    def _validate_environment(self) -> bool:
        """Validate the environment is ready"""
        issues = []
        
        # Check Python version
        if sys.version_info < (3, 8):
            issues.append("Python 3.8+ required")
        
        # Check required directories
        for dir_name in ['logs', 'data', 'checkpoints', 'output']:
            dir_path = Path(dir_name)
            if not dir_path.exists():
                dir_path.mkdir(parents=True)
                logger.debug(f"Created directory: {dir_name}")
        
        # Check config
        if not self.config:
            issues.append("Configuration not loaded")
        
        if issues:
            for issue in issues:
                logger.error(f"Validation issue: {issue}")
            return False
        
        logger.info("Environment validation passed")
        return True
    
    def _generate_report(self):
        """Generate discovery report"""
        if not self.orchestrator:
            return
        
        stats = self.orchestrator.get_statistics()
        
        print("\n" + "="*60)
        print("DISCOVERY REPORT")
        print("="*60)
        print(f"Start Time:           {stats.get('start_time', 'N/A')}")
        print(f"End Time:             {stats.get('end_time', 'N/A')}")
        print(f"Duration:             {stats.get('duration', 'N/A')}")
        print("-"*60)
        print(f"Data Sources:         {stats.get('data_sources', 0):,}")
        print(f"Tables Processed:     {stats.get('tables_processed', 0):,}")
        print(f"Rows Scanned:         {stats.get('rows_scanned', 0):,}")
        print("-"*60)
        print(f"Hosts Discovered:     {stats.get('hosts_discovered', 0):,}")
        print(f"Applications:         {stats.get('applications', 0):,}")
        print(f"Relationships:        {stats.get('relationships', 0):,}")
        print(f"Unique Environments:  {stats.get('environments', 0):,}")
        print(f"Unique Datacenters:   {stats.get('datacenters', 0):,}")
        print("-"*60)
        print(f"Classification Accuracy: {stats.get('classification_accuracy', 0):.2%}")
        print(f"Data Quality Score:      {stats.get('quality_score', 0):.2%}")
        print("="*60)
        print(f"\nCMDB Database: {self.config.get('output_database', 'cmdb.db')}")
        print(f"Query with: sqlite3 {self.config.get('output_database', 'cmdb.db')}")
        print("="*60)
    
    def _print_analysis_results(self, quality_report, anomalies, insights):
        """Print analysis results"""
        print("\n" + "="*60)
        print("CMDB ANALYSIS RESULTS")
        print("="*60)
        
        # Data quality
        print("\nData Quality:")
        print("-"*40)
        for metric, value in quality_report.items():
            print(f"  {metric:30s}: {value}")
        
        # Anomalies
        print(f"\nAnomalies Detected: {len(anomalies)}")
        print("-"*40)
        for i, anomaly in enumerate(anomalies[:10], 1):
            print(f"  {i}. {anomaly['type']}: {anomaly['description']}")
        
        # Insights
        print(f"\nKey Insights:")
        print("-"*40)
        for i, insight in enumerate(insights[:10], 1):
            print(f"  {i}. {insight}")
        
        print("="*60)

async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Smart CMDB Discovery System')
    parser.add_argument(
        '--mode',
        choices=['discover', 'analyze', 'demo'],
        default='discover',
        help='Operation mode'
    )
    parser.add_argument(
        '--config',
        default='config.json',
        help='Configuration file path'
    )
    parser.add_argument(
        '--no-resume',
        action='store_true',
        help='Start fresh without resuming from checkpoint'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    
    args = parser.parse_args()
    
    # Set debug level if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create system instance
    system = CMDBDiscoverySystem(args.config)
    
    # Disable resume if requested
    if args.no_resume:
        system.config['resume_enabled'] = False
    
    # Run the system
    success = await system.run(args.mode)
    
    if success:
        logger.info("✅ Discovery completed successfully!")
        sys.exit(0)
    else:
        logger.error("❌ Discovery failed")
        sys.exit(1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutdown requested")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)