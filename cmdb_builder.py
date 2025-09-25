import duckdb
import json
import asyncio
from typing import List, Dict, Any, Optional, Set
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class CMDBBuilder:
    
    def __init__(self, db_path: str = 'new_cmdb.db'):
        self.db_path = db_path
        self.conn = None
        self.dynamic_columns = set()
        
    async def initialize(self):
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._init_sync)
        
    def _init_sync(self):
        # Connect to DuckDB
        self.conn = duckdb.connect(self.db_path)
        self.conn.execute("SET memory_limit='8GB'")
        self.conn.execute("SET threads=8")
        self.conn.execute("SET wal_autocheckpoint='1GB'")
        logger.info(f"DuckDB initialized at {self.db_path}")
        
    async def create_hosts_table(self, columns: List[str]):
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._create_hosts_table_sync, columns)
        
    def _create_hosts_table_sync(self, columns: List[str]):
        # Drop existing table if it exists
        self.conn.execute("DROP TABLE IF EXISTS hosts")
        
        # Build CREATE TABLE statement
        column_defs = [
            "hostname VARCHAR PRIMARY KEY",
            "raw_forms TEXT",
            "occurrence_count INTEGER",
            "confidence FLOAT"
        ]
        
        # Add dynamic columns
        for col in columns[4:]:  # Skip the first 4 standard columns
            # Sanitize column name
            safe_col = self._sanitize_column_name(col)
            column_defs.append(f"{safe_col} VARCHAR")
            self.dynamic_columns.add(safe_col)
            
        create_sql = f"CREATE TABLE hosts ({', '.join(column_defs)})"
        
        self.conn.execute(create_sql)
        
        # Also create relationships table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS host_relationships (
                id INTEGER PRIMARY KEY,
                hostname VARCHAR,
                source_table VARCHAR,
                source_column VARCHAR,
                confidence FLOAT,
                discovered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (hostname) REFERENCES hosts(hostname)
            )
        """)
        
        # Create metadata table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS cmdb_metadata (
                key VARCHAR PRIMARY KEY,
                value TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        logger.info(f"Created hosts table with {len(columns)} columns")
        
    def _sanitize_column_name(self, column_name: str) -> str:
        import re
        safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', column_name)
        
        # Ensure it doesn't start with a number
        if safe_name and safe_name[0].isdigit():
            safe_name = 'col_' + safe_name
            
        # Truncate if too long
        if len(safe_name) > 64:
            safe_name = safe_name[:64]
            
        return safe_name.lower()
        
    async def bulk_insert(self, table: str, records: List[Dict]):
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._bulk_insert_sync, table, records)
        
    def _bulk_insert_sync(self, table: str, records: List[Dict]):
        if not records:
            return
            
        # Process in batches
        batch_size = 10000
        
        for i in range(0, len(records), batch_size):
            batch = records[i:i+batch_size]
            
            # Convert records to DataFrame-like format
            for record in batch:
                # Get columns and values
                columns = []
                values = []
                
                for col, val in record.items():
                    safe_col = self._sanitize_column_name(col)
                    columns.append(safe_col)
                    
                    # Handle different value types
                    if val is None:
                        values.append(None)
                    elif isinstance(val, (dict, list)):
                        values.append(json.dumps(val))
                    else:
                        values.append(str(val))
                        
                # Build INSERT statement
                placeholders = ', '.join(['?' for _ in values])
                columns_str = ', '.join(columns)
                
                insert_sql = f"""
                INSERT OR REPLACE INTO {table} ({columns_str})
                VALUES ({placeholders})
                """
                
                try:
                    self.conn.execute(insert_sql, values)
                except Exception as e:
                    logger.warning(f"Failed to insert record: {e}")
                    
            self.conn.commit()
            logger.info(f"Inserted batch of {len(batch)} records")
            
        # Update metadata
        self._update_metadata()
        
    def _update_metadata(self):
        # Get statistics
        host_count = self.conn.execute("SELECT COUNT(*) FROM hosts").fetchone()[0]
        
        metadata = {
            'total_hosts': host_count,
            'last_updated': datetime.now().isoformat(),
            'dynamic_columns': list(self.dynamic_columns)
        }
        
        for key, value in metadata.items():
            self.conn.execute("""
                INSERT OR REPLACE INTO cmdb_metadata (key, value)
                VALUES (?, ?)
            """, [key, json.dumps(value) if isinstance(value, (list, dict)) else str(value)])
            
        self.conn.commit()
        
    async def create_indexes(self, columns: List[str]):
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._create_indexes_sync, columns)
        
    def _create_indexes_sync(self, columns: List[str]):
        for column in columns:
            safe_col = self._sanitize_column_name(column)
            
            # Check if column exists
            try:
                # Create index
                index_name = f"idx_{safe_col}"
                self.conn.execute(f"CREATE INDEX IF NOT EXISTS {index_name} ON hosts({safe_col})")
                logger.debug(f"Created index on {safe_col}")
            except Exception as e:
                logger.debug(f"Could not create index on {safe_col}: {e}")
                
        self.conn.commit()
        logger.info(f"Created indexes on {len(columns)} columns")
        
    def query(self, sql: str) -> List[Dict]:
        try:
            result = self.conn.execute(sql)
            columns = [desc[0] for desc in result.description]
            
            rows = []
            for row in result.fetchall():
                rows.append(dict(zip(columns, row)))
                
            return rows
            
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return []
            
    def get_statistics(self) -> Dict[str, Any]:
        stats = {}
        
        try:
            # Total hosts
            stats['total_hosts'] = self.conn.execute(
                "SELECT COUNT(*) FROM hosts"
            ).fetchone()[0]
            
            # Hosts with high confidence
            stats['high_confidence_hosts'] = self.conn.execute(
                "SELECT COUNT(*) FROM hosts WHERE confidence > 0.8"
            ).fetchone()[0]
            
            # Average occurrence count
            result = self.conn.execute(
                "SELECT AVG(occurrence_count) FROM hosts"
            ).fetchone()
            stats['avg_occurrences'] = result[0] if result else 0
            
            # Total relationships
            stats['total_relationships'] = self.conn.execute(
                "SELECT COUNT(*) FROM host_relationships"
            ).fetchone()[0]
            
            # Dynamic columns
            stats['dynamic_columns'] = len(self.dynamic_columns)
            
            # Get metadata
            metadata = self.conn.execute(
                "SELECT key, value FROM cmdb_metadata"
            ).fetchall()
            
            for key, value in metadata:
                try:
                    stats[f"metadata_{key}"] = json.loads(value)
                except:
                    stats[f"metadata_{key}"] = value
                    
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            
        return stats
        
    def export_to_csv(self, output_path: str):
        try:
            self.conn.execute(f"""
                COPY hosts TO '{output_path}' (FORMAT CSV, HEADER)
            """)
            logger.info(f"Exported CMDB to {output_path}")
        except Exception as e:
            logger.error(f"Export failed: {e}")
            
    def close(self):
        if self.conn:
            self.conn.close()
            logger.info("DuckDB connection closed")