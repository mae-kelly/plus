#!/usr/bin/env python3
"""
BigQuery CMDB Discovery System
Scans BigQuery projects to automatically discover and build a CMDB of your infrastructure
"""

import asyncio
import json
import logging
import argparse
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List

from core.bigquery_scanner import BigQueryScanner
from core.cmdb_discovery import CMDBDiscovery
from core.cmdb_builder import CMDBBuilder
from utils.logger import setup_logging

logger = setup_logging('bigquery_cmdb')

class BigQueryCMDBSystem:
    """Main system to scan BigQuery and build CMDB"""
    
    def __init__(self, config_path: str = 'config.json'):
        self.config = self.load_config(config_path)
        self.scanner = BigQueryScanner(self.config)
        self.discovery = CMDBDiscovery(self.config)
        self.builder = CMDBBuilder(self.config.get('output_database', 'cmdb.db'))
        self.stats = {
            'start_time': datetime.now(),
            'projects_scanned': 0,
            'datasets_scanned': 0,
            'tables_scanned': 0,
            'rows_processed': 0,
            'hosts_discovered': 0,
            'relationships_found': 0
        }
    
    def load_config(self, config_path: str) -> Dict:
        """Load configuration"""
        path = Path(config_path)
        if not path.exists():
            logger.error(f"Config file not found: {config_path}")
            self.create_default_config(path)
            logger.info(f"Created default config at {config_path}")
            logger.info("Please edit it with your BigQuery projects and credentials")
            sys.exit(1)
        
        with open(path) as f:
            config = json.load(f)
        
        # Validate BigQuery configuration
        if not config.get('bigquery', {}).get('projects'):
            logger.error("No BigQuery projects configured!")
            logger.error("Edit config.json and add your project IDs under 'bigquery.projects'")
            sys.exit(1)
        
        return config
    
    def create_default_config(self, path: Path):
        """Create default configuration for BigQuery scanning"""
        config = {
            "bigquery": {
                "projects": ["your-project-id-1", "your-project-id-2"],
                "credentials_path": "gcp_credentials.json",
                "datasets_filter": [],  # Leave empty to scan all datasets
                "tables_filter": [],    # Leave empty to scan all tables
                "sample_percent": 10,   # Sample 10% of large tables
                "max_rows_per_table": 100000
            },
            "discovery": {
                "hostname_patterns": [
                    {"column_pattern": ".*host.*", "confidence": 0.9},
                    {"column_pattern": ".*server.*", "confidence": 0.8},
                    {"column_pattern": ".*instance.*", "confidence": 0.8},
                    {"column_pattern": ".*node.*", "confidence": 0.7},
                    {"column_pattern": ".*machine.*", "confidence": 0.7},
                    {"column_pattern": ".*device.*", "confidence": 0.6}
                ],
                "ip_patterns": [
                    {"column_pattern": ".*ip.*address.*", "confidence": 0.95},
                    {"column_pattern": ".*ip.*", "confidence": 0.8},
                    {"column_pattern": ".*addr.*", "confidence": 0.6}
                ],
                "environment_patterns": [
                    {"column_pattern": ".*env.*", "confidence": 0.9},
                    {"column_pattern": ".*environment.*", "confidence": 0.95},
                    {"column_pattern": ".*stage.*", "confidence": 0.7}
                ],
                "application_patterns": [
                    {"column_pattern": ".*app.*", "confidence": 0.8},
                    {"column_pattern": ".*service.*", "confidence": 0.8},
                    {"column_pattern": ".*application.*", "confidence": 0.9}
                ]
            },
            "output_database": "bigquery_cmdb.db",
            "export": {
                "csv": true,
                "json": true,
                "output_dir": "output"
            },
            "logging": {
                "level": "INFO",
                "file": "logs/bigquery_cmdb.log"
            }
        }
        
        with open(path, 'w') as f:
            json.dump(config, f, indent=2)
    
    async def run(self):
        """Main execution"""
        logger.info("="*60)
        logger.info("BigQuery CMDB Discovery System")
        logger.info("="*60)
        
        # Step 1: Scan BigQuery
        logger.info("\n📊 Step 1: Scanning BigQuery projects...")
        bigquery_data = await self.scanner.scan_all_projects()
        
        if not bigquery_data:
            logger.error("No data found in BigQuery! Check your configuration and credentials.")
            return False
        
        self.stats['projects_scanned'] = len(self.scanner.projects_scanned)
        self.stats['datasets_scanned'] = self.scanner.datasets_scanned
        self.stats['tables_scanned'] = self.scanner.tables_scanned
        self.stats['rows_processed'] = self.scanner.rows_processed
        
        logger.info(f"✅ Scanned {self.stats['tables_scanned']} tables, {self.stats['rows_processed']:,} rows")
        
        # Step 2: Discover hosts and infrastructure
        logger.info("\n🔍 Step 2: Discovering infrastructure from BigQuery data...")
        discovered_hosts = await self.discovery.discover_hosts(bigquery_data)
        
        self.stats['hosts_discovered'] = len(discovered_hosts)
        logger.info(f"✅ Discovered {self.stats['hosts_discovered']} hosts/devices")
        
        # Step 3: Find relationships
        logger.info("\n🔗 Step 3: Mapping relationships...")
        relationships = await self.discovery.find_relationships(discovered_hosts)
        
        self.stats['relationships_found'] = len(relationships)
        logger.info(f"✅ Found {self.stats['relationships_found']} relationships")
        
        # Step 4: Build CMDB
        logger.info("\n🏗️ Step 4: Building CMDB database...")
        await self.builder.build(discovered_hosts, relationships)
        logger.info(f"✅ CMDB created at {self.config['output_database']}")
        
        # Step 5: Export results
        if self.config.get('export', {}).get('csv'):
            await self.export_results(discovered_hosts, 'csv')
        if self.config.get('export', {}).get('json'):
            await self.export_results(discovered_hosts, 'json')
        
        # Print summary
        self.print_summary()
        
        return True
    
    async def export_results(self, hosts: Dict, format: str):
        """Export discovered hosts"""
        output_dir = Path(self.config.get('export', {}).get('output_dir', 'output'))
        output_dir.mkdir(exist_ok=True)
        
        if format == 'csv':
            import csv
            output_file = output_dir / f"cmdb_hosts_{datetime.now():%Y%m%d_%H%M%S}.csv"
            
            with open(output_file, 'w', newline='') as f:
                if hosts:
                    fieldnames = set()
                    for host in hosts.values():
                        fieldnames.update(host.keys())
                    
                    writer = csv.DictWriter(f, fieldnames=sorted(fieldnames))
                    writer.writeheader()
                    writer.writerows(hosts.values())
            
            logger.info(f"📁 Exported to {output_file}")
        
        elif format == 'json':
            output_file = output_dir / f"cmdb_hosts_{datetime.now():%Y%m%d_%H%M%S}.json"
            
            with open(output_file, 'w') as f:
                json.dump(hosts, f, indent=2, default=str)
            
            logger.info(f"📁 Exported to {output_file}")
    
    def print_summary(self):
        """Print discovery summary"""
        duration = (datetime.now() - self.stats['start_time']).total_seconds()
        
        print("\n" + "="*60)
        print("BIGQUERY CMDB DISCOVERY SUMMARY")
        print("="*60)
        print(f"Duration:             {duration:.1f} seconds")
        print(f"Projects Scanned:     {self.stats['projects_scanned']}")
        print(f"Datasets Scanned:     {self.stats['datasets_scanned']}")
        print(f"Tables Scanned:       {self.stats['tables_scanned']}")
        print(f"Rows Processed:       {self.stats['rows_processed']:,}")
        print("-"*60)
        print(f"Hosts Discovered:     {self.stats['hosts_discovered']}")
        print(f"Relationships Found:  {self.stats['relationships_found']}")
        print("-"*60)
        print(f"CMDB Database:        {self.config['output_database']}")
        print(f"Query with:           sqlite3 {self.config['output_database']}")
        print("="*60)
        
        # Show example queries
        print("\nExample queries to explore your CMDB:")
        print("  SELECT * FROM hosts LIMIT 10;")
        print("  SELECT environment, COUNT(*) FROM hosts GROUP BY environment;")
        print("  SELECT h1.hostname, h2.hostname, r.relationship_type")
        print("    FROM relationships r")
        print("    JOIN hosts h1 ON r.source_id = h1.id")
        print("    JOIN hosts h2 ON r.target_id = h2.id;")
        print("="*60)

async def main():
    parser = argparse.ArgumentParser(
        description='BigQuery CMDB Discovery - Scan BigQuery to build infrastructure CMDB'
    )
    parser.add_argument('--config', default='config.json', help='Configuration file')
    parser.add_argument('--projects', nargs='+', help='Override BigQuery projects to scan')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    system = BigQueryCMDBSystem(args.config)
    
    # Override projects if specified
    if args.projects:
        system.config['bigquery']['projects'] = args.projects
    
    success = await system.run()
    
    if success:
        logger.info("✅ BigQuery CMDB Discovery completed successfully!")
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