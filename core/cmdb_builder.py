"""
CMDB Builder - Builds and manages the CMDB database
"""

import sqlite3
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class CMDBBuilder:
    """Builds and manages the CMDB database"""
    
    def __init__(self, db_path: str = 'cmdb.db'):
        self.db_path = db_path
        self.conn = None
        self.initialized = False
    
    async def initialize(self):
        """Initialize database connection"""
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row
            
            # Enable foreign keys
            self.conn.execute("PRAGMA foreign_keys = ON")
            
            # Optimize for performance
            self.conn.execute("PRAGMA journal_mode = WAL")
            self.conn.execute("PRAGMA synchronous = NORMAL")
            self.conn.execute("PRAGMA cache_size = 10000")
            self.conn.execute("PRAGMA temp_store = MEMORY")
            
            self.initialized = True
            logger.info(f"CMDB database initialized at {self.db_path}")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    async def create_schema(self, additional_columns: List[str]):
        """Create database schema"""
        if not self.initialized:
            await self.initialize()
        
        try:
            # Drop existing tables
            self.conn.execute("DROP TABLE IF EXISTS relationships")
            self.conn.execute("DROP TABLE IF EXISTS entities")
            self.conn.execute("DROP TABLE IF EXISTS attributes")
            self.conn.execute("DROP TABLE IF EXISTS statistics")
            self.conn.execute("DROP TABLE IF EXISTS audit_log")
            
            # Create entities table
            entity_columns = [
                "entity_id INTEGER PRIMARY KEY AUTOINCREMENT",
                "hostname TEXT UNIQUE NOT NULL",
                "entity_type TEXT",
                "sub_type TEXT",
                "environment TEXT",
                "datacenter TEXT",
                "os_type TEXT",
                "application TEXT",
                "owner TEXT",
                "criticality TEXT",
                "confidence REAL",
                "classification_method TEXT",
                "tags TEXT",  # JSON array
                "first_seen TIMESTAMP",
                "last_seen TIMESTAMP",
                "last_modified TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
                "is_active BOOLEAN DEFAULT 1"
            ]
            
            # Add dynamic columns
            for col in additional_columns[:50]:  # Limit to 50 additional columns
                safe_col = self._sanitize_column_name(col)
                if safe_col and safe_col not in ['entity_id', 'hostname']:
                    entity_columns.append(f"{safe_col} TEXT")
            
            create_entities_sql = f"""
            CREATE TABLE entities (
                {', '.join(entity_columns)}
            )"""
            
            self.conn.execute(create_entities_sql)
            
            # Create attributes table for overflow/dynamic attributes
            self.conn.execute("""
            CREATE TABLE attributes (
                attribute_id INTEGER PRIMARY KEY AUTOINCREMENT,
                entity_id INTEGER NOT NULL,
                attribute_name TEXT NOT NULL,
                attribute_value TEXT,
                attribute_type TEXT,
                confidence REAL,
                source TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (entity_id) REFERENCES entities(entity_id) ON DELETE CASCADE,
                UNIQUE(entity_id, attribute_name)
            )""")
            
            # Create relationships table
            self.conn.execute("""
            CREATE TABLE relationships (
                relationship_id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_entity_id INTEGER NOT NULL,
                target_entity_id INTEGER NOT NULL,
                relationship_type TEXT NOT NULL,
                confidence REAL,
                via_attribute TEXT,
                discovered_at TIMESTAMP,
                last_verified TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_active BOOLEAN DEFAULT 1,
                FOREIGN KEY (source_entity_id) REFERENCES entities(entity_id) ON DELETE CASCADE,
                FOREIGN KEY (target_entity_id) REFERENCES entities(entity_id) ON DELETE CASCADE,
                UNIQUE(source_entity_id, target_entity_id, relationship_type)
            )""")
            
            # Create statistics table
            self.conn.execute("""
            CREATE TABLE statistics (
                stat_id INTEGER PRIMARY KEY AUTOINCREMENT,
                stat_name TEXT UNIQUE NOT NULL,
                stat_value TEXT,
                stat_type TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )""")
            
            # Create audit log table
            self.conn.execute("""
            CREATE TABLE audit_log (
                log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                entity_id INTEGER,
                action TEXT NOT NULL,
                details TEXT,
                user TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (entity_id) REFERENCES entities(entity_id) ON DELETE SET NULL
            )""")
            
            self.conn.commit()
            logger.info("Database schema created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create schema: {e}")
            self.conn.rollback()
            raise
    
    async def insert_entities(self, entities: List[Dict]):
        """Insert entities into database"""
        if not self.initialized:
            await self.initialize()
        
        inserted_count = 0
        entity_id_map = {}
        
        for entity in entities:
            try:
                # Prepare core fields
                hostname = entity.get('hostname')
                if not hostname:
                    continue
                
                # Insert main entity
                cursor = self.conn.execute("""
                INSERT OR REPLACE INTO entities (
                    hostname, entity_type, sub_type, environment, datacenter,
                    os_type, application, owner, criticality, confidence,
                    classification_method, tags, first_seen, last_seen
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    hostname,
                    entity.get('entity_type', 'unknown'),
                    entity.get('sub_type', 'unknown'),
                    entity.get('environment', 'unknown'),
                    entity.get('datacenter', 'unknown'),
                    entity.get('os_type', 'unknown'),
                    entity.get('application', 'unknown'),
                    entity.get('owner', 'unknown'),
                    entity.get('criticality', 'medium'),
                    entity.get('confidence', 0.5),
                    entity.get('classification', {}).get('classification_method', 'unknown'),
                    json.dumps(entity.get('classification', {}).get('tags', [])),
                    entity.get('first_seen', datetime.now().isoformat()),
                    entity.get('last_seen', datetime.now().isoformat())
                ))
                
                entity_id = cursor.lastrowid
                entity_id_map[hostname] = entity_id
                
                # Insert additional attributes
                attributes = entity.get('attributes', {})
                for attr_name, attr_values in attributes.items():
                    if isinstance(attr_values, list):
                        attr_value = json.dumps(attr_values[:10])  # Limit array size
                    else:
                        attr_value = str(attr_values)
                    
                    self.conn.execute("""
                    INSERT OR REPLACE INTO attributes (
                        entity_id, attribute_name, attribute_value, attribute_type
                    ) VALUES (?, ?, ?, ?)
                    """, (
                        entity_id,
                        attr_name,
                        attr_value,
                        self._infer_attribute_type(attr_value)
                    ))
                
                inserted_count += 1
                
            except Exception as e:
                logger.error(f"Failed to insert entity {hostname}: {e}")
        
        self.conn.commit()
        logger.info(f"Inserted {inserted_count} entities")
        
        return entity_id_map
    
    async def insert_relationships(self, relationships: List[Dict]):
        """Insert relationships into database"""
        if not self.initialized:
            await self.initialize()
        
        # First get entity IDs
        cursor = self.conn.execute("SELECT hostname, entity_id FROM entities")
        entity_map = {row['hostname']: row['entity_id'] for row in cursor.fetchall()}
        
        inserted_count = 0
        
        for rel in relationships:
            try:
                source = rel.get('source')
                target = rel.get('target')
                
                if source not in entity_map or target not in entity_map:
                    continue
                
                source_id = entity_map[source]
                target_id = entity_map[target]
                
                self.conn.execute("""
                INSERT OR REPLACE INTO relationships (
                    source_entity_id, target_entity_id, relationship_type,
                    confidence, via_attribute, discovered_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    source_id,
                    target_id,
                    rel.get('type', 'related'),
                    rel.get('confidence', 0.5),
                    rel.get('via_attribute'),
                    rel.get('discovered_at', datetime.now().isoformat())
                ))
                
                inserted_count += 1
                
            except Exception as e:
                logger.error(f"Failed to insert relationship: {e}")
        
        self.conn.commit()
        logger.info(f"Inserted {inserted_count} relationships")
    
    async def create_indexes(self):
        """Create database indexes for performance"""
        if not self.initialized:
            return
        
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_entities_hostname ON entities(hostname)",
            "CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type)",
            "CREATE INDEX IF NOT EXISTS idx_entities_env ON entities(environment)",
            "CREATE INDEX IF NOT EXISTS idx_entities_dc ON entities(datacenter)",
            "CREATE INDEX IF NOT EXISTS idx_entities_app ON entities(application)",
            "CREATE INDEX IF NOT EXISTS idx_entities_owner ON entities(owner)",
            "CREATE INDEX IF NOT EXISTS idx_entities_criticality ON entities(criticality)",
            "CREATE INDEX IF NOT EXISTS idx_entities_confidence ON entities(confidence)",
            "CREATE INDEX IF NOT EXISTS idx_entities_active ON entities(is_active)",
            
            "CREATE INDEX IF NOT EXISTS idx_attributes_entity ON attributes(entity_id)",
            "CREATE INDEX IF NOT EXISTS idx_attributes_name ON attributes(attribute_name)",
            
            "CREATE INDEX IF NOT EXISTS idx_relationships_source ON relationships(source_entity_id)",
            "CREATE INDEX IF NOT EXISTS idx_relationships_target ON relationships(target_entity_id)",
            "CREATE INDEX IF NOT EXISTS idx_relationships_type ON relationships(relationship_type)",
            
            "CREATE INDEX IF NOT EXISTS idx_audit_entity ON audit_log(entity_id)",
            "CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp)"
        ]
        
        for index_sql in indexes:
            try:
                self.conn.execute(index_sql)
            except Exception as e:
                logger.warning(f"Failed to create index: {e}")
        
        self.conn.commit()
        logger.info("Database indexes created")
    
    async def store_statistics(self, stats: Dict):
        """Store statistics in database"""
        if not self.initialized:
            return
        
        for stat_name, stat_value in stats.items():
            try:
                if isinstance(stat_value, (dict, list)):
                    stat_value = json.dumps(stat_value)
                else:
                    stat_value = str(stat_value)
                
                self.conn.execute("""
                INSERT OR REPLACE INTO statistics (stat_name, stat_value, stat_type)
                VALUES (?, ?, ?)
                """, (
                    stat_name,
                    stat_value,
                    type(stats[stat_name]).__name__
                ))
            except Exception as e:
                logger.error(f"Failed to store statistic {stat_name}: {e}")
        
        self.conn.commit()
        logger.info("Statistics stored")
    
    async def load_all_data(self) -> Dict:
        """Load all data from CMDB"""
        if not self.initialized:
            await self.initialize()
        
        try:
            # Load entities
            cursor = self.conn.execute("""
            SELECT * FROM entities WHERE is_active = 1
            """)
            entities = [dict(row) for row in cursor.fetchall()]
            
            # Load attributes
            cursor = self.conn.execute("""
            SELECT * FROM attributes
            """)
            attributes = [dict(row) for row in cursor.fetchall()]
            
            # Load relationships
            cursor = self.conn.execute("""
            SELECT * FROM relationships WHERE is_active = 1
            """)
            relationships = [dict(row) for row in cursor.fetchall()]
            
            # Load statistics
            cursor = self.conn.execute("""
            SELECT * FROM statistics
            """)
            statistics = {row['stat_name']: row['stat_value'] for row in cursor.fetchall()}
            
            return {
                'entities': entities,
                'attributes': attributes,
                'relationships': relationships,
                'statistics': statistics
            }
            
        except Exception as e:
            logger.error(f"Failed to load CMDB data: {e}")
            return {}
    
    def _sanitize_column_name(self, name: str) -> str:
        """Sanitize column name for SQL"""
        import re
        # Remove special characters and spaces
        safe = re.sub(r'[^a-zA-Z0-9_]', '_', name)
        # Ensure it doesn't start with a number
        if safe and safe[0].isdigit():
            safe = 'col_' + safe
        # Limit length
        if len(safe) > 64:
            safe = safe[:64]
        return safe.lower()
    
    def _infer_attribute_type(self, value: str) -> str:
        """Infer attribute type from value"""
        try:
            # Try to parse as JSON
            json.loads(value)
            return 'json'
        except:
            pass
        
        # Check for numeric
        try:
            float(value)
            return 'numeric'
        except:
            pass
        
        # Check for date patterns
        import re
        if re.match(r'\d{4}-\d{2}-\d{2}', value):
            return 'date'
        
        # Check for boolean
        if value.lower() in ['true', 'false', 'yes', 'no', '1', '0']:
            return 'boolean'
        
        return 'text'
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            self.initialized = False
            logger.info("Database connection closed")