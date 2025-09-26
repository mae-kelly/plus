"""
Data Scanner - Scans multiple data sources for infrastructure information
"""

import csv
import json
import sqlite3
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
import re

logger = logging.getLogger(__name__)

class DataScanner:
    """Scans various data sources for infrastructure data"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.scan_results = []
    
    async def scan_csv_files(self, file_paths: List[str]) -> List[Dict]:
        """Scan CSV files for data"""
        results = []
        
        for file_path in file_paths:
            path = Path(file_path)
            if not path.exists():
                logger.warning(f"CSV file not found: {file_path}")
                continue
            
            try:
                with open(path, 'r', encoding='utf-8-sig') as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                    
                    if rows:
                        # Analyze columns
                        columns = {}
                        for col in reader.fieldnames:
                            col_values = [row[col] for row in rows if row.get(col)]
                            columns[col] = self._analyze_column(col, col_values)
                        
                        results.append({
                            'type': 'csv',
                            'name': path.name,
                            'path': str(path),
                            'tables': [{
                                'name': path.stem,
                                'rows': rows,
                                'columns': columns,
                                'row_count': len(rows)
                            }]
                        })
                        
                        logger.info(f"Scanned CSV {path.name}: {len(rows)} rows")
                        
            except Exception as e:
                logger.error(f"Error scanning CSV {file_path}: {e}")
        
        return results
    
    async def scan_json_files(self, file_paths: List[str]) -> List[Dict]:
        """Scan JSON files for data"""
        results = []
        
        for file_path in file_paths:
            path = Path(file_path)
            if not path.exists():
                logger.warning(f"JSON file not found: {file_path}")
                continue
            
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    # Handle different JSON structures
                    tables = []
                    
                    if isinstance(data, list):
                        # Array of objects
                        if data and isinstance(data[0], dict):
                            columns = {}
                            for key in data[0].keys():
                                col_values = [item.get(key) for item in data]
                                columns[key] = self._analyze_column(key, col_values)
                            
                            tables.append({
                                'name': path.stem,
                                'rows': data,
                                'columns': columns,
                                'row_count': len(data)
                            })
                    
                    elif isinstance(data, dict):
                        # Object with potential nested arrays
                        for key, value in data.items():
                            if isinstance(value, list) and value and isinstance(value[0], dict):
                                columns = {}
                                for col in value[0].keys():
                                    col_values = [item.get(col) for item in value]
                                    columns[col] = self._analyze_column(col, col_values)
                                
                                tables.append({
                                    'name': key,
                                    'rows': value,
                                    'columns': columns,
                                    'row_count': len(value)
                                })
                    
                    if tables:
                        results.append({
                            'type': 'json',
                            'name': path.name,
                            'path': str(path),
                            'tables': tables
                        })
                        
                        logger.info(f"Scanned JSON {path.name}: {len(tables)} tables")
                        
            except Exception as e:
                logger.error(f"Error scanning JSON {file_path}: {e}")
        
        return results
    
    async def scan_databases(self, db_configs: List[Dict]) -> List[Dict]:
        """Scan databases for data"""
        results = []
        
        for db_config in db_configs:
            db_type = db_config.get('type', 'sqlite')
            
            if db_type == 'sqlite':
                result = await self._scan_sqlite(db_config)
                if result:
                    results.append(result)
            else:
                logger.warning(f"Unsupported database type: {db_type}")
        
        return results
    
    async def _scan_sqlite(self, db_config: Dict) -> Optional[Dict]:
        """Scan SQLite database"""
        db_path = db_config.get('path')
        if not db_path or not Path(db_path).exists():
            logger.warning(f"SQLite database not found: {db_path}")
            return None
        
        try:
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Get list of tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            table_names = [row[0] for row in cursor.fetchall()]
            
            tables = []
            for table_name in table_names:
                # Skip system tables
                if table_name.startswith('sqlite_'):
                    continue
                
                # Get table data
                cursor.execute(f"SELECT * FROM {table_name} LIMIT 10000")
                rows = [dict(row) for row in cursor.fetchall()]
                
                if rows:
                    # Analyze columns
                    columns = {}
                    for col in rows[0].keys():
                        col_values = [row[col] for row in rows if row.get(col)]
                        columns[col] = self._analyze_column(col, col_values)
                    
                    tables.append({
                        'name': table_name,
                        'rows': rows,
                        'columns': columns,
                        'row_count': len(rows)
                    })
            
            conn.close()
            
            if tables:
                return {
                    'type': 'sqlite',
                    'name': Path(db_path).name,
                    'path': db_path,
                    'tables': tables
                }
                
        except Exception as e:
            logger.error(f"Error scanning SQLite {db_path}: {e}")
        
        return None
    
    async def scan_apis(self, api_configs: List[Dict]) -> List[Dict]:
        """Scan REST APIs for data"""
        results = []
        
        # Note: This is a placeholder for API scanning
        # In production, you would implement actual API calls here
        
        for api_config in api_configs:
            logger.info(f"API scanning not implemented: {api_config.get('name')}")
        
        return results
    
    async def scan_default_locations(self) -> List[Dict]:
        """Scan default locations for data files"""
        results = []
        
        # Default directories to scan
        default_dirs = [
            Path('data'),
            Path('input'),
            Path('samples'),
            Path.cwd()
        ]
        
        for directory in default_dirs:
            if not directory.exists():
                continue
            
            # Scan for CSV files
            csv_files = list(directory.glob('*.csv'))
            if csv_files:
                csv_results = await self.scan_csv_files([str(f) for f in csv_files[:10]])
                results.extend(csv_results)
            
            # Scan for JSON files
            json_files = list(directory.glob('*.json'))
            # Exclude config files
            json_files = [f for f in json_files if 'config' not in f.name.lower()]
            if json_files:
                json_results = await self.scan_json_files([str(f) for f in json_files[:10]])
                results.extend(json_results)
            
            # Scan for SQLite databases
            db_files = list(directory.glob('*.db')) + list(directory.glob('*.sqlite'))
            if db_files:
                db_configs = [{'type': 'sqlite', 'path': str(f)} for f in db_files[:5]]
                db_results = await self.scan_databases(db_configs)
                results.extend(db_results)
        
        # If no data found, generate demo data
        if not results:
            logger.info("No data sources found, generating demo data...")
            results = self._generate_demo_data()
        
        return results
    
    def _analyze_column(self, column_name: str, values: List[Any]) -> Dict:
        """Analyze a column's characteristics"""
        if not values:
            return {
                'type': 'unknown',
                'unique_count': 0,
                'unique_ratio': 0,
                'null_count': 0,
                'null_ratio': 0,
                'sample_values': []
            }
        
        # Filter out None values
        non_null_values = [v for v in values if v is not None and str(v).strip()]
        null_count = len(values) - len(non_null_values)
        
        if not non_null_values:
            return {
                'type': 'empty',
                'unique_count': 0,
                'unique_ratio': 0,
                'null_count': null_count,
                'null_ratio': 1.0,
                'sample_values': []
            }
        
        # Determine data type
        data_type = self._infer_data_type(non_null_values)
        
        # Calculate statistics
        unique_values = set(str(v) for v in non_null_values)
        unique_count = len(unique_values)
        unique_ratio = unique_count / len(non_null_values)
        
        # Get sample values
        sample_values = list(unique_values)[:10]
        
        return {
            'type': data_type,
            'unique_count': unique_count,
            'unique_ratio': unique_ratio,
            'null_count': null_count,
            'null_ratio': null_count / len(values),
            'sample_values': sample_values,
            'total_count': len(values)
        }
    
    def _infer_data_type(self, values: List[Any]) -> str:
        """Infer the data type of values"""
        if not values:
            return 'unknown'
        
        # Sample first 100 values
        sample = values[:100]
        
        # Check for patterns
        numeric_count = 0
        date_count = 0
        ip_count = 0
        email_count = 0
        hostname_count = 0
        
        for value in sample:
            str_val = str(value)
            
            # Check for numeric
            try:
                float(str_val)
                numeric_count += 1
            except:
                pass
            
            # Check for date patterns
            if re.match(r'\d{4}-\d{2}-\d{2}', str_val):
                date_count += 1
            
            # Check for IP address
            if re.match(r'^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$', str_val):
                ip_count += 1
            
            # Check for email
            if '@' in str_val and '.' in str_val:
                email_count += 1
            
            # Check for hostname pattern
            if re.match(r'^[a-zA-Z0-9][a-zA-Z0-9\-\.]*[a-zA-Z0-9]$', str_val):
                hostname_count += 1
        
        # Determine type based on counts
        sample_size = len(sample)
        
        if numeric_count > sample_size * 0.9:
            return 'numeric'
        elif date_count > sample_size * 0.8:
            return 'date'
        elif ip_count > sample_size * 0.8:
            return 'ip_address'
        elif email_count > sample_size * 0.8:
            return 'email'
        elif hostname_count > sample_size * 0.6:
            return 'hostname'
        else:
            return 'text'
    
    def _generate_demo_data(self) -> List[Dict]:
        """Generate demo data for testing"""
        demo_rows = []
        
        # Generate demo servers
        environments = ['prod', 'staging', 'dev', 'test']
        datacenters = ['us-east-1', 'us-west-2', 'eu-west-1']
        apps = ['web', 'api', 'database', 'cache', 'queue']
        
        for i in range(50):
            env = environments[i % len(environments)]
            dc = datacenters[i % len(datacenters)]
            app = apps[i % len(apps)]
            
            demo_rows.append({
                'hostname': f"{app}-{env}-{i:03d}.example.com",
                'ip_address': f"10.{i//100}.{i%100}.{i%256}",
                'environment': env,
                'datacenter': dc,
                'application': app,
                'os': 'linux' if i % 3 else 'windows',
                'cpu_cores': 4 * (1 + i % 4),
                'memory_gb': 8 * (1 + i % 8),
                'owner': f"team-{i % 5}",
                'status': 'active' if i % 10 else 'maintenance'
            })
        
        # Analyze columns
        columns = {}
        for col in demo_rows[0].keys():
            col_values = [row[col] for row in demo_rows]
            columns[col] = self._analyze_column(col, col_values)
        
        return [{
            'type': 'demo',
            'name': 'demo_infrastructure',
            'path': 'generated',
            'tables': [{
                'name': 'servers',
                'rows': demo_rows,
                'columns': columns,
                'row_count': len(demo_rows)
            }]
        }]