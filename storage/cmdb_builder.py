import duckdb
import json
from typing import List, Dict, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class CMDBBuilder:
    def __init__(self, db_path: str = 'cmdb.db'):
        self.db_path = db_path
        self.conn = None
        self.table_created = False
    
    async def initialize(self):
        self.conn = duckdb.connect(self.db_path)
        
        self.conn.execute("SET memory_limit='8GB'")
        self.conn.execute("SET threads=8")
        self.conn.execute("SET wal_autocheckpoint='1GB'")
        
        logger.info(f"DuckDB initialized at {self.db_path}")
    
    async def create_schema(self, columns: List[Dict]):
        self.conn.execute("DROP TABLE IF EXISTS hosts")
        self.conn.execute("DROP TABLE IF EXISTS host_relationships")
        self.conn.execute("DROP TABLE IF EXISTS metadata")
        
        column_definitions = [
            "hostname VARCHAR PRIMARY KEY",
            "raw_forms TEXT",
            "occurrence_count INTEGER",
            "confidence FLOAT",
            "quality_score FLOAT",
            "discovered_at TIMESTAMP"
        ]
        
        for col_info in columns:
            col_name = self._sanitize_column_name(col_info['name'])
            col_type = self._map_column_type(col_info.get('type', 'unknown'))
            column_definitions.append(f"{col_name} {col_type}")
        
        create_sql = f"CREATE TABLE hosts ({', '.join(column_definitions)})"
        self.conn.execute(create_sql)
        
        self.conn.execute("""
            CREATE TABLE host_relationships (
                id INTEGER PRIMARY KEY,
                source_host VARCHAR,
                target_host VARCHAR,
                relationship_type VARCHAR,
                confidence FLOAT,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (source_host) REFERENCES hosts(hostname),
                FOREIGN KEY (target_host) REFERENCES hosts(hostname)
            )
        """)
        
        self.conn.execute("""
            CREATE TABLE metadata (
                key VARCHAR PRIMARY KEY,
                value TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        self.table_created = True
        logger.info(f"Created schema with {len(columns)} columns")
    
    async def insert_hosts(self, hosts: List[Dict]):
        if not self.table_created:
            return
        
        for host in hosts:
            try:
                columns = []
                values = []
                placeholders = []
                
                for key, value in host.items():
                    col_name = self._sanitize_column_name(key)
                    columns.append(col_name)
                    
                    if value is None:
                        values.append(None)
                    elif isinstance(value, (dict, list)):
                        values.append(json.dumps(value))
                    elif isinstance(value, datetime):
                        values.append(value.isoformat())
                    else:
                        values.append(str(value))
                    
                    placeholders.append('?')
                
                insert_sql = f"""
                    INSERT OR REPLACE INTO hosts ({', '.join(columns)})
                    VALUES ({', '.join(placeholders)})
                """
                
                self.conn.execute(insert_sql, values)
                
            except Exception as e:
                logger.error(f"Failed to insert host: {e}")
        
        self.conn.commit()
        logger.info(f"Inserted {len(hosts)} hosts")
    
    async def insert_relationships(self, relationships: List[Dict]):
        for rel in relationships:
            try:
                self.conn.execute("""
                    INSERT INTO host_relationships 
                    (source_host, target_host, relationship_type, confidence, metadata)
                    VALUES (?, ?, ?, ?, ?)
                """, [
                    rel['source'],
                    rel['target'],
                    rel['type'],
                    rel.get('confidence', 0.5),
                    json.dumps(rel.get('metadata', {}))
                ])
            except Exception as e:
                logger.error(f"Failed to insert relationship: {e}")
        
        self.conn.commit()
    
    async def create_indexes(self):
        indexes = [
            "CREATE INDEX idx_confidence ON hosts(confidence)",
            "CREATE INDEX idx_quality ON hosts(quality_score)",
            "CREATE INDEX idx_occurrence ON hosts(occurrence_count)",
            "CREATE INDEX idx_discovered ON hosts(discovered_at)",
            "CREATE INDEX idx_rel_source ON host_relationships(source_host)",
            "CREATE INDEX idx_rel_target ON host_relationships(target_host)",
            "CREATE INDEX idx_rel_type ON host_relationships(relationship_type)"
        ]
        
        for index_sql in indexes:
            try:
                self.conn.execute(index_sql)
            except Exception as e:
                logger.debug(f"Index creation: {e}")
        
        self.conn.commit()
        logger.info("Indexes created")
    
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
    
    def get_statistics(self) -> Dict:
        stats = {}
        
        try:
            stats['total_hosts'] = self.conn.execute(
                "SELECT COUNT(*) FROM hosts"
            ).fetchone()[0]
            
            stats['high_confidence'] = self.conn.execute(
                "SELECT COUNT(*) FROM hosts WHERE confidence > 0.8"
            ).fetchone()[0]
            
            stats['total_relationships'] = self.conn.execute(
                "SELECT COUNT(*) FROM host_relationships"
            ).fetchone()[0]
            
            stats['avg_quality_score'] = self.conn.execute(
                "SELECT AVG(quality_score) FROM hosts"
            ).fetchone()[0]
            
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
        
        return stats
    
    def export_to_parquet(self, output_path: str):
        try:
            self.conn.execute(f"""
                COPY hosts TO '{output_path}' (FORMAT PARQUET)
            """)
            logger.info(f"Exported to {output_path}")
        except Exception as e:
            logger.error(f"Export failed: {e}")
    
    def _sanitize_column_name(self, name: str) -> str:
        import re
        safe = re.sub(r'[^a-zA-Z0-9_]', '_', name)
        
        if safe and safe[0].isdigit():
            safe = 'col_' + safe
        
        if len(safe) > 64:
            safe = safe[:64]
        
        return safe.lower()
    
    def _map_column_type(self, semantic_type: str) -> str:
        type_mapping = {
            'hostname': 'VARCHAR',
            'ip_address': 'VARCHAR',
            'email': 'VARCHAR',
            'timestamp': 'TIMESTAMP',
            'date': 'DATE',
            'amount': 'DOUBLE',
            'percentage': 'DOUBLE',
            'numeric_id': 'BIGINT',
            'boolean': 'BOOLEAN',
            'json': 'TEXT'
        }
        
        return type_mapping.get(semantic_type, 'VARCHAR')
    
    def close(self):
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")