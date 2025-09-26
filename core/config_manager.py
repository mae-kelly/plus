"""
Configuration Manager - Manages system configuration, validation, and defaults
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
import copy

logger = logging.getLogger(__name__)

class ConfigManager:
    """Manages system configuration with validation and defaults"""
    
    def __init__(self, config_path: str = 'config.json'):
        self.config_path = Path(config_path)
        self.config = {}
        self.default_config = self._get_default_config()
        self.config_schema = self._get_config_schema()
        self.environment_overrides = self._load_environment_overrides()
        
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file with validation"""
        if not self.config_path.exists():
            logger.info(f"Config file not found at {self.config_path}, creating default configuration")
            self.create_default_config()
        
        try:
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
                logger.info(f"Configuration loaded from {self.config_path}")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in config file: {e}")
            logger.info("Using default configuration")
            self.config = self.default_config.copy()
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            self.config = self.default_config.copy()
        
        # Merge with defaults (fill missing values)
        self.config = self._merge_with_defaults(self.config)
        
        # Apply environment variable overrides
        self.config = self._apply_environment_overrides(self.config)
        
        # Validate configuration
        validation_errors = self.validate_config(self.config)
        if validation_errors:
            logger.warning(f"Configuration validation warnings: {validation_errors}")
        
        # Expand paths
        self.config = self._expand_paths(self.config)
        
        # Log configuration summary
        self._log_config_summary()
        
        return self.config
    
    def save_config(self, config: Optional[Dict] = None):
        """Save configuration to file"""
        if config:
            self.config = config
        
        try:
            # Create backup of existing config
            if self.config_path.exists():
                backup_path = self.config_path.with_suffix('.bak')
                self.config_path.rename(backup_path)
                logger.info(f"Backed up existing config to {backup_path}")
            
            # Save new config
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2, sort_keys=True)
                logger.info(f"Configuration saved to {self.config_path}")
                
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
            # Restore backup if save failed
            backup_path = self.config_path.with_suffix('.bak')
            if backup_path.exists():
                backup_path.rename(self.config_path)
                logger.info("Restored backup configuration")
    
    def create_default_config(self):
        """Create default configuration file"""
        self.config = self.default_config.copy()
        self.save_config()
        logger.info(f"Created default configuration at {self.config_path}")
    
    def validate_config(self, config: Dict) -> List[str]:
        """Validate configuration against schema"""
        errors = []
        
        # Check required fields
        required_fields = self.config_schema.get('required', [])
        for field in required_fields:
            if field not in config:
                errors.append(f"Missing required field: {field}")
        
        # Validate data sources
        data_sources = config.get('data_sources', {})
        if not any(data_sources.values()):
            logger.info("No data sources configured, will scan default locations")
        
        # Validate file paths
        for source_type, paths in data_sources.items():
            if source_type in ['csv_files', 'json_files']:
                for path in paths:
                    if not Path(path).exists():
                        errors.append(f"File not found: {path}")
        
        # Validate database configurations
        for db_config in data_sources.get('databases', []):
            if 'type' not in db_config:
                errors.append("Database configuration missing 'type' field")
            if db_config.get('type') == 'sqlite' and 'path' not in db_config:
                errors.append("SQLite database configuration missing 'path' field")
        
        # Validate thresholds (should be between 0 and 1)
        discovery = config.get('discovery', {})
        thresholds = [
            'hostname_confidence_threshold',
            'relationship_confidence_threshold'
        ]
        for threshold in thresholds:
            value = discovery.get(threshold)
            if value is not None:
                if not (0 <= value <= 1):
                    errors.append(f"{threshold} must be between 0 and 1, got {value}")
        
        # Validate numeric fields
        if config.get('max_workers', 1) < 1:
            errors.append("max_workers must be at least 1")
        
        if config.get('batch_size', 1) < 1:
            errors.append("batch_size must be at least 1")
        
        # Validate output database
        output_db = config.get('output_database')
        if output_db:
            output_path = Path(output_db)
            if output_path.exists() and not output_path.is_file():
                errors.append(f"Output database path is not a file: {output_db}")
        
        # Validate logging configuration
        log_config = config.get('logging', {})
        valid_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if log_config.get('level') not in valid_log_levels:
            errors.append(f"Invalid log level: {log_config.get('level')}")
        
        return errors
    
    def get_config_value(self, key_path: str, default: Any = None) -> Any:
        """Get configuration value by dot-notation path"""
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
                if value is None:
                    return default
            else:
                return default
        
        return value
    
    def set_config_value(self, key_path: str, value: Any):
        """Set configuration value by dot-notation path"""
        keys = key_path.split('.')
        config = self.config
        
        # Navigate to parent
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        # Set the value
        config[keys[-1]] = value
        logger.debug(f"Set config {key_path} = {value}")
    
    def merge_configs(self, *configs: Dict) -> Dict:
        """Merge multiple configurations"""
        result = {}
        for config in configs:
            result = self._deep_merge(result, config)
        return result
    
    def export_config(self, output_path: str, format: str = 'json'):
        """Export configuration in various formats"""
        output_path = Path(output_path)
        
        if format == 'json':
            with open(output_path, 'w') as f:
                json.dump(self.config, f, indent=2)
        elif format == 'env':
            # Export as environment variables
            with open(output_path, 'w') as f:
                self._write_env_vars(self.config, f)
        elif format == 'yaml':
            # Export as YAML (if PyYAML available)
            try:
                import yaml
                with open(output_path, 'w') as f:
                    yaml.dump(self.config, f, default_flow_style=False)
            except ImportError:
                logger.error("PyYAML not installed, cannot export to YAML")
        else:
            logger.error(f"Unsupported export format: {format}")
        
        logger.info(f"Configuration exported to {output_path}")
    
    def get_profile(self, profile_name: str) -> Dict:
        """Load a configuration profile"""
        profile_path = Path(f'profiles/{profile_name}.json')
        
        if not profile_path.exists():
            logger.warning(f"Profile not found: {profile_name}")
            return {}
        
        try:
            with open(profile_path, 'r') as f:
                profile_config = json.load(f)
                logger.info(f"Loaded profile: {profile_name}")
                return profile_config
        except Exception as e:
            logger.error(f"Failed to load profile {profile_name}: {e}")
            return {}
    
    def apply_profile(self, profile_name: str):
        """Apply a configuration profile"""
        profile_config = self.get_profile(profile_name)
        if profile_config:
            self.config = self._deep_merge(self.config, profile_config)
            logger.info(f"Applied profile: {profile_name}")
    
    # Private helper methods
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "data_sources": {
                "csv_files": [],
                "json_files": [],
                "databases": [],
                "apis": []
            },
            "discovery": {
                "hostname_confidence_threshold": 0.6,
                "relationship_confidence_threshold": 0.5,
                "max_rows_per_source": 100000,
                "sample_size": 1000,
                "parallel_workers": 4,
                "batch_size": 100
            },
            "classification": {
                "confidence_threshold": 0.7,
                "use_cache": True,
                "cache_ttl": 3600
            },
            "output_database": "cmdb.db",
            "resume_enabled": True,
            "checkpoint_interval": 1000,
            "demo_hosts": 100,
            "demo_apps": 20,
            "logging": {
                "level": "INFO",
                "file": "logs/cmdb.log",
                "max_size": "10MB",
                "backup_count": 5,
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            },
            "performance": {
                "max_workers": 4,
                "batch_size": 64,
                "memory_limit": "4GB",
                "enable_profiling": False
            },
            "quality": {
                "enable_quality_checks": True,
                "anomaly_detection": True,
                "insight_generation": True,
                "quality_threshold": 0.7
            },
            "export": {
                "formats": ["json", "csv", "parquet"],
                "output_directory": "output",
                "compress": False
            },
            "notifications": {
                "enabled": False,
                "email": {
                    "smtp_server": "",
                    "smtp_port": 587,
                    "from_address": "",
                    "to_addresses": []
                },
                "slack": {
                    "webhook_url": ""
                }
            },
            "advanced": {
                "entity_resolution": {
                    "enabled": True,
                    "similarity_threshold": 0.8,
                    "fuzzy_matching": True
                },
                "relationship_discovery": {
                    "max_depth": 3,
                    "inference_enabled": True,
                    "pattern_matching": True
                },
                "data_enrichment": {
                    "enabled": True,
                    "dns_lookup": False,
                    "reverse_dns": False,
                    "geo_location": False
                }
            }
        }
    
    def _get_config_schema(self) -> Dict:
        """Get configuration schema for validation"""
        return {
            "required": ["data_sources", "discovery", "output_database"],
            "properties": {
                "data_sources": {
                    "type": "object",
                    "properties": {
                        "csv_files": {"type": "array"},
                        "json_files": {"type": "array"},
                        "databases": {"type": "array"},
                        "apis": {"type": "array"}
                    }
                },
                "discovery": {
                    "type": "object",
                    "properties": {
                        "hostname_confidence_threshold": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1
                        },
                        "relationship_confidence_threshold": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1
                        }
                    }
                }
            }
        }
    
    def _merge_with_defaults(self, config: Dict) -> Dict:
        """Merge configuration with defaults"""
        return self._deep_merge(self.default_config, config)
    
    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """Deep merge two dictionaries"""
        result = copy.deepcopy(base)
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = copy.deepcopy(value)
        
        return result
    
    def _load_environment_overrides(self) -> Dict:
        """Load configuration overrides from environment variables"""
        overrides = {}
        
        # Map environment variables to config paths
        env_mappings = {
            'CMDB_OUTPUT_DATABASE': 'output_database',
            'CMDB_LOG_LEVEL': 'logging.level',
            'CMDB_MAX_WORKERS': 'performance.max_workers',
            'CMDB_BATCH_SIZE': 'performance.batch_size',
            'CMDB_RESUME_ENABLED': 'resume_enabled',
            'CMDB_DEMO_HOSTS': 'demo_hosts',
            'CMDB_QUALITY_THRESHOLD': 'quality.quality_threshold'
        }
        
        for env_var, config_path in env_mappings.items():
            value = os.environ.get(env_var)
            if value:
                # Convert to appropriate type
                if value.lower() in ['true', 'false']:
                    value = value.lower() == 'true'
                elif value.isdigit():
                    value = int(value)
                elif '.' in value and value.replace('.', '').isdigit():
                    value = float(value)
                
                # Store override
                keys = config_path.split('.')
                current = overrides
                for key in keys[:-1]:
                    if key not in current:
                        current[key] = {}
                    current = current[key]
                current[keys[-1]] = value
                
                logger.debug(f"Environment override: {config_path} = {value}")
        
        return overrides
    
    def _apply_environment_overrides(self, config: Dict) -> Dict:
        """Apply environment variable overrides to configuration"""
        if self.environment_overrides:
            return self._deep_merge(config, self.environment_overrides)
        return config
    
    def _expand_paths(self, config: Dict) -> Dict:
        """Expand relative paths to absolute paths"""
        result = copy.deepcopy(config)
        
        # Expand data source paths
        if 'data_sources' in result:
            for source_type in ['csv_files', 'json_files']:
                if source_type in result['data_sources']:
                    result['data_sources'][source_type] = [
                        str(Path(path).resolve()) 
                        for path in result['data_sources'][source_type]
                    ]
            
            # Expand database paths
            for db_config in result['data_sources'].get('databases', []):
                if 'path' in db_config:
                    db_config['path'] = str(Path(db_config['path']).resolve())
        
        # Expand output database path
        if 'output_database' in result:
            result['output_database'] = str(Path(result['output_database']).resolve())
        
        # Expand log file path
        if 'logging' in result and 'file' in result['logging']:
            result['logging']['file'] = str(Path(result['logging']['file']).resolve())
        
        # Expand export directory
        if 'export' in result and 'output_directory' in result['export']:
            result['export']['output_directory'] = str(Path(result['export']['output_directory']).resolve())
        
        return result
    
    def _write_env_vars(self, config: Dict, file_handle, prefix: str = 'CMDB'):
        """Write configuration as environment variables"""
        def flatten(d: Dict, parent_key: str = ''):
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}_{k.upper()}" if parent_key else f"{prefix}_{k.upper()}"
                if isinstance(v, dict):
                    items.extend(flatten(v, new_key).items())
                else:
                    items.append((new_key, v))
            return dict(items)
        
        flat_config = flatten(config)
        for key, value in flat_config.items():
            if isinstance(value, bool):
                value = 'true' if value else 'false'
            elif isinstance(value, list):
                value = ','.join(str(v) for v in value)
            file_handle.write(f"{key}={value}\n")
    
    def _log_config_summary(self):
        """Log configuration summary"""
        summary = []
        
        # Data sources
        data_sources = self.config.get('data_sources', {})
        source_count = sum(len(v) for v in data_sources.values() if isinstance(v, list))
        summary.append(f"Data sources: {source_count}")
        
        # Discovery settings
        discovery = self.config.get('discovery', {})
        summary.append(f"Hostname threshold: {discovery.get('hostname_confidence_threshold', 'N/A')}")
        summary.append(f"Relationship threshold: {discovery.get('relationship_confidence_threshold', 'N/A')}")
        
        # Output
        summary.append(f"Output database: {self.config.get('output_database', 'N/A')}")
        
        # Features
        features = []
        if self.config.get('resume_enabled'):
            features.append('resume')
        if self.config.get('quality', {}).get('anomaly_detection'):
            features.append('anomaly detection')
        if self.config.get('advanced', {}).get('entity_resolution', {}).get('enabled'):
            features.append('entity resolution')
        
        if features:
            summary.append(f"Features: {', '.join(features)}")
        
        logger.info(f"Configuration loaded: {' | '.join(summary)}")
    
    def get_runtime_info(self) -> Dict:
        """Get runtime configuration information"""
        return {
            'config_path': str(self.config_path),
            'config_exists': self.config_path.exists(),
            'environment_overrides': bool(self.environment_overrides),
            'validation_errors': self.validate_config(self.config),
            'loaded_at': datetime.now().isoformat(),
            'summary': {
                'data_sources': sum(
                    len(v) for v in self.config.get('data_sources', {}).values() 
                    if isinstance(v, list)
                ),
                'output_database': self.config.get('output_database'),
                'resume_enabled': self.config.get('resume_enabled'),
                'quality_checks': self.config.get('quality', {}).get('enable_quality_checks')
            }
        }