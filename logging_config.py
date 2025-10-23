"""
Centralized logging configuration for SingN'Seek.
Configures logging to both file (with rotation) and Elasticsearch.
"""

import logging
import os
import yaml
import json
from pathlib import Path
from typing import Optional
from datetime import datetime

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Try to import Elasticsearch for direct logging
try:
    from elasticsearch import Elasticsearch
    ELASTICSEARCH_AVAILABLE = True
except ImportError:
    ELASTICSEARCH_AVAILABLE = False


# Load configuration
CONFIG_PATH = Path(__file__).parent / "config.yaml"
with open(CONFIG_PATH, 'r') as f:
    CONFIG = yaml.safe_load(f)


class ElasticsearchHandler(logging.Handler):
    """Custom handler to send logs to Elasticsearch."""
    
    def __init__(self, es_client, index_name, index_frequency='daily'):
        super().__init__()
        self.es_client = es_client
        self.base_index_name = index_name
        self.index_frequency = index_frequency
    
    def get_index_name(self):
        """Generate index name with time suffix."""
        now = datetime.now()
        if self.index_frequency == 'daily':
            suffix = now.strftime('%Y.%m.%d')
        elif self.index_frequency == 'weekly':
            suffix = now.strftime('%Y.%W')
        elif self.index_frequency == 'monthly':
            suffix = now.strftime('%Y.%m')
        elif self.index_frequency == 'yearly':
            suffix = now.strftime('%Y')
        else:
            suffix = now.strftime('%Y.%m.%d')
        
        return f"{self.base_index_name}-{suffix}"
    
    def emit(self, record):
        """Send log record to Elasticsearch."""
        try:
            # Format the log record - only send the message
            log_entry = {
                '@timestamp': datetime.utcnow().isoformat(),
                'message': record.getMessage(),  # Only the actual log message
                'level': record.levelname,
                'logger': record.name,
                'function': record.funcName,
                'line': record.lineno,
                'module': record.module,
                'application': 'singnseek'
            }
            
            # Add exception info if present
            if record.exc_info:
                log_entry['exception'] = self.formatException(record.exc_info)
            
            # Index the document
            index_name = self.get_index_name()
            self.es_client.index(
                index=index_name,
                document=log_entry
            )
        except Exception:
            # Don't let logging errors break the application
            self.handleError(record)


def setup_logging(
    name: str = None,
    log_level: Optional[str] = None
) -> logging.Logger:
    """
    Configure logging for the application.
    
    Args:
        name: Logger name (default: None for root logger)
        log_level: Log level (default: from config or INFO)
    
    Returns:
        Configured logger instance
    """
    # Use root logger to configure all loggers
    logger = logging.getLogger(name)
    
    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger
    
    # Set log level
    log_config = CONFIG.get('logging', {})
    if log_level is None:
        log_level = log_config.get('level', 'INFO')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        fmt='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # ==================================================
    # Console Handler (for ERROR and above only)
    # ==================================================
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.ERROR)  # Only show errors in console
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)
    
    # ==================================================
    # Elasticsearch Handler
    # ==================================================
    if ELASTICSEARCH_AVAILABLE and log_config.get('elasticsearch', {}).get('enabled', False):
        try:
            es_config = log_config['elasticsearch']
            
            # Get Elasticsearch connection details
            connection_type = os.getenv('ELASTICSEARCH_CONNECTION_TYPE', 'local')
            api_key = os.getenv('ELASTICSEARCH_API_KEY')
            
            if api_key:
                # Create Elasticsearch client
                es_client = None
                
                if connection_type == 'cloud' or os.getenv('ELASTICSEARCH_CLOUD_ID'):
                    # Cloud connection
                    cloud_id = os.getenv('ELASTICSEARCH_CLOUD_ID')
                    if cloud_id:
                        es_client = Elasticsearch(
                            cloud_id=cloud_id,
                            api_key=api_key,
                            request_timeout=30
                        )
                else:
                    # Local connection
                    host = os.getenv('ELASTICSEARCH_HOST', 'localhost')
                    port = int(os.getenv('ELASTICSEARCH_PORT', '9200'))
                    scheme = os.getenv('ELASTICSEARCH_SCHEME', 'http')
                    
                    es_client = Elasticsearch(
                        [f"{scheme}://{host}:{port}"],
                        api_key=api_key,
                        request_timeout=30
                    )
                
                # Add custom Elasticsearch handler
                if es_client and es_client.ping():
                    es_handler = ElasticsearchHandler(
                        es_client=es_client,
                        index_name=es_config.get('index_name', 'singnseek-logs'),
                        index_frequency=es_config.get('index_frequency', 'daily')
                    )
                    es_handler.setLevel(logging.INFO)
                    es_handler.setFormatter(detailed_formatter)
                    logger.addHandler(es_handler)
                    logger.info("Elasticsearch logging handler configured successfully")
                
        except Exception as e:
            # Log warning but don't fail if Elasticsearch logging can't be set up
            print(f"Warning: Could not set up Elasticsearch logging: {e}")
            import traceback
            traceback.print_exc()
    
    # Don't prevent propagation - let child loggers use these handlers
    logger.propagate = False
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the standard configuration.
    All loggers will use the handlers configured in setup_logging().
    
    Args:
        name: Logger name (typically __name__)
    
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)
