"""
Logger - Advanced logging configuration with rotation, formatting, and multiple handlers
"""

import logging
import sys
import os
from pathlib import Path
from datetime import datetime
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from typing import Optional, Dict, Any
import json
import traceback

class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for console output"""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'       # Reset
    }
    
    def __init__(self, fmt=None, datefmt=None, use_colors=True):
        super().__init__(fmt, datefmt)
        self.use_colors = use_colors and sys.stdout.isatty()
    
    def format(self, record):
        if self.use_colors:
            # Add color to level name
            levelname = record.levelname
            if levelname in self.COLORS:
                record.levelname = f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"
        
        # Format the message
        result = super().format(record)
        
        # Reset level name for next handler
        record.levelname = record.levelname.replace(self.COLORS.get(record.levelname, ''), '').replace(self.COLORS['RESET'], '')
        
        return result

class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def format(self, record):
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'message': record.getMessage(),
            'process': record.process,
            'thread': record.thread
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'created', 'filename', 'funcName',
                          'levelname', 'levelno', 'lineno', 'module', 'msecs',
                          'message', 'pathname', 'process', 'processName', 'relativeCreated',
                          'thread', 'threadName', 'exc_info', 'exc_text', 'stack_info']:
                log_data[key] = value
        
        return json.dumps(log_data)

class CMDBLogger:
    """Advanced logger setup for CMDB Discovery System"""
    
    def __init__(self):
        self.loggers = {}
        self.handlers = {}
        self.log_dir = Path('logs')
        self.log_dir.mkdir(exist_ok=True)
    
    def setup_logger(
        self,
        name: str = 'cmdb',
        level: str = 'INFO',
        log_file: Optional[str] = None,
        console: bool = True,
        json_format: bool = False,
        max_bytes: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5,
        rotation_type: str = 'size'  # 'size', 'time', or 'both'
    ) -> logging.Logger:
        """Setup a logger with specified configuration"""
        
        # Get or create logger
        logger = logging.getLogger(name)
        
        # Avoid duplicate handlers
        if name in self.loggers:
            return self.loggers[name]
        
        logger.setLevel(getattr(logging, level.upper()))
        logger.handlers = []
        
        # Console handler
        if console:
            console_handler = self._create_console_handler(level, json_format)
            logger.addHandler(console_handler)
            self.handlers[f"{name}_console"] = console_handler
        
        # File handler
        if log_file:
            file_handler = self._create_file_handler(
                log_file, level, json_format, max_bytes, backup_count, rotation_type
            )
            logger.addHandler(file_handler)
            self.handlers[f"{name}_file"] = file_handler
        
        # Error file handler (separate file for errors)
        error_log_file = self.log_dir / f'{name}_errors.log'
        error_handler = self._create_error_handler(error_log_file, json_format)
        logger.addHandler(error_handler)
        self.handlers[f"{name}_errors"] = error_handler
        
        # Prevent propagation to root logger
        logger.propagate = False
        
        # Store logger
        self.loggers[name] = logger
        
        # Log initial message
        logger.info(f"Logger '{name}' initialized - Level: {level}")
        
        return logger
    
    def _create_console_handler(self, level: str, json_format: bool) -> logging.Handler:
        """Create console handler with formatting"""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))
        
        if json_format:
            formatter = JSONFormatter()
        else:
            # Use colored formatter for console
            fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            formatter = ColoredFormatter(fmt, datefmt='%Y-%m-%d %H:%M:%S')
        
        console_handler.setFormatter(formatter)
        return console_handler
    
    def _create_file_handler(
        self,
        log_file: str,
        level: str,
        json_format: bool,
        max_bytes: int,
        backup_count: int,
        rotation_type: str
    ) -> logging.Handler:
        """Create file handler with rotation"""
        
        log_path = self.log_dir / log_file
        
        if rotation_type == 'time':
            # Rotate daily
            file_handler = TimedRotatingFileHandler(
                log_path,
                when='midnight',
                interval=1,
                backupCount=backup_count,
                encoding='utf-8'
            )
            file_handler.suffix = '%Y%m%d'
        elif rotation_type == 'both':
            # Custom handler that does both size and time rotation
            file_handler = RotatingFileHandler(
                log_path,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding='utf-8'
            )
        else:  # size
            file_handler = RotatingFileHandler(
                log_path,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding='utf-8'
            )
        
        file_handler.setLevel(getattr(logging, level.upper()))
        
        if json_format:
            formatter = JSONFormatter()
        else:
            fmt = '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
            formatter = logging.Formatter(fmt, datefmt='%Y-%m-%d %H:%M:%S')
        
        file_handler.setFormatter(formatter)
        return file_handler
    
    def _create_error_handler(self, log_file: Path, json_format: bool) -> logging.Handler:
        """Create handler specifically for errors"""
        error_handler = RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        
        if json_format:
            formatter = JSONFormatter()
        else:
            # Detailed format for errors
            fmt = (
                '%(asctime)s - %(name)s - %(levelname)s - '
                '%(pathname)s:%(lineno)d - %(funcName)s() - '
                '%(message)s'
            )
            formatter = logging.Formatter(fmt, datefmt='%Y-%m-%d %H:%M:%S')
        
        error_handler.setFormatter(formatter)
        return error_handler
    
    def get_logger(self, name: str) -> Optional[logging.Logger]:
        """Get an existing logger"""
        return self.loggers.get(name)
    
    def set_level(self, name: str, level: str):
        """Change logger level dynamically"""
        logger = self.loggers.get(name)
        if logger:
            logger.setLevel(getattr(logging, level.upper()))
            for handler in logger.handlers:
                handler.setLevel(getattr(logging, level.upper()))
            logger.info(f"Logger level changed to {level}")
    
    def add_file_handler(self, name: str, log_file: str):
        """Add a file handler to existing logger"""
        logger = self.loggers.get(name)
        if logger:
            handler = self._create_file_handler(
                log_file, 'INFO', False, 10*1024*1024, 5, 'size'
            )
            logger.addHandler(handler)
            self.handlers[f"{name}_{log_file}"] = handler
    
    def close_all(self):
        """Close all handlers and loggers"""
        for handler in self.handlers.values():
            handler.close()
        self.handlers.clear()
        self.loggers.clear()

class LogContext:
    """Context manager for temporary logging changes"""
    
    def __init__(self, logger_name: str, level: str = None, handler: logging.Handler = None):
        self.logger = logging.getLogger(logger_name)
        self.original_level = self.logger.level
        self.original_handlers = self.logger.handlers.copy()
        self.temp_level = getattr(logging, level.upper()) if level else None
        self.temp_handler = handler
    
    def __enter__(self):
        if self.temp_level:
            self.logger.setLevel(self.temp_level)
        if self.temp_handler:
            self.logger.addHandler(self.temp_handler)
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.setLevel(self.original_level)
        if self.temp_handler and self.temp_handler in self.logger.handlers:
            self.logger.removeHandler(self.temp_handler)

class PerformanceLogger:
    """Logger for performance metrics"""
    
    def __init__(self, logger_name: str = 'performance'):
        self.logger = logging.getLogger(logger_name)
        self.metrics = {}
    
    def log_timing(self, operation: str, duration: float, **kwargs):
        """Log operation timing"""
        self.logger.info(
            f"Performance: {operation}",
            extra={
                'operation': operation,
                'duration_ms': duration * 1000,
                'metrics': kwargs
            }
        )
        
        # Store metrics
        if operation not in self.metrics:
            self.metrics[operation] = []
        self.metrics[operation].append({
            'timestamp': datetime.now(),
            'duration': duration,
            **kwargs
        })
    
    def log_resource_usage(self, operation: str, cpu_percent: float, memory_mb: float):
        """Log resource usage"""
        self.logger.info(
            f"Resources: {operation}",
            extra={
                'operation': operation,
                'cpu_percent': cpu_percent,
                'memory_mb': memory_mb
            }
        )
    
    def get_statistics(self, operation: str = None) -> Dict:
        """Get performance statistics"""
        if operation:
            data = self.metrics.get(operation, [])
        else:
            data = []
            for op_data in self.metrics.values():
                data.extend(op_data)
        
        if not data:
            return {}
        
        durations = [d['duration'] for d in data]
        
        return {
            'count': len(durations),
            'mean': sum(durations) / len(durations),
            'min': min(durations),
            'max': max(durations),
            'total': sum(durations)
        }

# Singleton instance
_cmdb_logger = CMDBLogger()

def setup_logging(
    name: str = 'cmdb',
    level: str = None,
    config: Dict[str, Any] = None
) -> logging.Logger:
    """
    Main function to setup logging for the application
    
    Args:
        name: Logger name
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        config: Optional configuration dictionary
    
    Returns:
        Configured logger instance
    """
    
    # Use environment variable if no level specified
    if not level:
        level = os.environ.get('LOG_LEVEL', 'INFO')
    
    # Default configuration
    default_config = {
        'log_file': f'{name}.log',
        'console': True,
        'json_format': os.environ.get('LOG_FORMAT') == 'json',
        'max_bytes': 10 * 1024 * 1024,
        'backup_count': 5,
        'rotation_type': 'size'
    }
    
    # Merge with provided config
    if config:
        default_config.update(config)
    
    # Setup logger
    logger = _cmdb_logger.setup_logger(
        name=name,
        level=level,
        **default_config
    )
    
    # Add performance logger if debug mode
    if level == 'DEBUG':
        perf_logger = PerformanceLogger(f'{name}.performance')
        logger.perf = perf_logger  # Attach to main logger
    
    return logger

def get_logger(name: str) -> logging.Logger:
    """Get an existing logger or create a new one"""
    logger = _cmdb_logger.get_logger(name)
    if not logger:
        logger = setup_logging(name)
    return logger

def set_global_level(level: str):
    """Set level for all loggers"""
    for name in _cmdb_logger.loggers:
        _cmdb_logger.set_level(name, level)

def log_exception(logger: logging.Logger, exc: Exception, context: str = ""):
    """Log exception with full traceback"""
    logger.error(
        f"Exception in {context}: {str(exc)}",
        exc_info=True,
        extra={
            'exception_type': type(exc).__name__,
            'exception_message': str(exc),
            'traceback': traceback.format_exc(),
            'context': context
        }
    )

def create_audit_logger(name: str = 'audit') -> logging.Logger:
    """Create a logger specifically for audit trails"""
    audit_logger = setup_logging(
        name=name,
        level='INFO',
        config={
            'log_file': 'audit.log',
            'console': False,
            'json_format': True,
            'rotation_type': 'time'
        }
    )
    return audit_logger

# Decorators for logging
def log_execution(logger: logging.Logger = None):
    """Decorator to log function execution"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            nonlocal logger
            if not logger:
                logger = get_logger(func.__module__)
            
            logger.debug(f"Executing {func.__name__}")
            start_time = datetime.now()
            
            try:
                result = func(*args, **kwargs)
                duration = (datetime.now() - start_time).total_seconds()
                logger.debug(f"Completed {func.__name__} in {duration:.3f}s")
                return result
            except Exception as e:
                duration = (datetime.now() - start_time).total_seconds()
                logger.error(f"Failed {func.__name__} after {duration:.3f}s: {str(e)}")
                raise
        
        return wrapper
    return decorator

def log_performance(logger: logging.Logger = None):
    """Decorator to log performance metrics"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            nonlocal logger
            if not logger:
                logger = get_logger(func.__module__)
            
            start_time = datetime.now()
            
            # Memory tracking (if psutil available)
            try:
                import psutil
                process = psutil.Process()
                start_memory = process.memory_info().rss / 1024 / 1024  # MB
            except ImportError:
                start_memory = 0
            
            result = func(*args, **kwargs)
            
            duration = (datetime.now() - start_time).total_seconds()
            
            # Log performance
            if hasattr(logger, 'perf'):
                logger.perf.log_timing(func.__name__, duration)
            else:
                logger.info(f"Performance: {func.__name__} took {duration:.3f}s")
            
            return result
        
        return wrapper
    return decorator

# Initialize default logger
default_logger = setup_logging('cmdb')