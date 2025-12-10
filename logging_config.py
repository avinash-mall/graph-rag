"""
Standardized Structured Logging Configuration

This module provides uniform structured logging across all modules with:
- Consistent log level, message schema, and context fields
- JSON-formatted logs for production (optional)
- Standard context fields for tracing and debugging
"""

import logging
import sys
import json
import os
from typing import Dict, Any, Optional
from datetime import datetime, timezone
from contextvars import ContextVar

from config import get_config

# Context variable for request tracking
request_id_context: ContextVar[Optional[str]] = ContextVar('request_id', default=None)
operation_context: ContextVar[Optional[str]] = ContextVar('operation', default=None)


class StructuredFormatter(logging.Formatter):
    """
    Structured log formatter that adds consistent context fields.
    
    Formats logs as JSON in production, or human-readable in development.
    """
    
    def __init__(self, use_json: bool = False, include_context: bool = True):
        super().__init__()
        self.use_json = use_json
        self.include_context = include_context
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with structured fields"""
        # Extract standard fields
        log_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        # Add context from ContextVar
        if self.include_context:
            request_id = request_id_context.get()
            operation = operation_context.get()
            if request_id:
                log_data["request_id"] = request_id
            if operation:
                log_data["operation"] = operation
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Add any extra fields from the log record
        if hasattr(record, 'extra'):
            log_data.update(record.extra)
        
        # Extract extra fields from log record (set via extra= parameter)
        for key, value in record.__dict__.items():
            if key not in [
                'name', 'msg', 'args', 'created', 'filename', 'funcName',
                'levelname', 'levelno', 'lineno', 'module', 'msecs',
                'message', 'pathname', 'process', 'processName', 'relativeCreated',
                'thread', 'threadName', 'exc_info', 'exc_text', 'stack_info',
                'getMessage'
            ]:
                if not key.startswith('_'):
                    log_data[key] = value
        
        if self.use_json:
            return json.dumps(log_data, default=str)
        else:
            # Human-readable format
            parts = [
                f"[{log_data['timestamp']}]",
                f"{log_data['level']:8s}",
                f"{log_data['logger']:20s}",
                log_data['message']
            ]
            
            # Add context fields
            if 'request_id' in log_data:
                parts.insert(-1, f"[req:{log_data['request_id']}]")
            if 'operation' in log_data:
                parts.insert(-1, f"[op:{log_data['operation']}]")
            
            # Add extra fields
            extra_fields = {k: v for k, v in log_data.items() 
                          if k not in ['timestamp', 'level', 'logger', 'message', 'request_id', 'operation']}
            if extra_fields:
                parts.append(f"| {json.dumps(extra_fields, default=str)}")
            
            return " ".join(parts)


class ContextLoggerAdapter(logging.LoggerAdapter):
    """
    Logger adapter that automatically includes context in log records.
    """
    
    def __init__(self, logger: logging.Logger, context: Optional[Dict[str, Any]] = None):
        super().__init__(logger, context or {})
    
    def process(self, msg: str, kwargs: Dict[str, Any]) -> tuple:
        """Add context to log record"""
        # Merge context from adapter and extra from call
        extra = kwargs.get('extra', {})
        if self.extra:
            if extra:
                extra = {**self.extra, **extra}
            else:
                extra = self.extra.copy()
        
        # Add context variables
        request_id = request_id_context.get()
        operation = operation_context.get()
        if request_id:
            extra['request_id'] = request_id
        if operation:
            extra['operation'] = operation
        
        kwargs['extra'] = extra
        return msg, kwargs


def setup_logging(
    log_level: Optional[str] = None,
    use_json: bool = False,
    log_to_file: bool = False,
    log_file: str = "graph_rag.log"
) -> None:
    """
    Setup standardized logging configuration.
    
    Args:
        log_level: Logging level (defaults to config)
        use_json: Use JSON formatting (defaults to config/env)
        log_to_file: Write logs to file
        log_file: Log file path
    """
    if log_level is None:
        cfg = get_config()
        log_level = cfg.app.log_level
    
    # Check environment for JSON logging
    if not use_json:
        use_json = os.getenv("LOG_JSON", "false").lower() == "true"
    
    # Create formatter
    formatter = StructuredFormatter(use_json=use_json)
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_to_file or os.getenv("LOG_TO_FILE", "false").lower() == "true":
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Set levels for third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("neo4j").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def get_logger(name: str, context: Optional[Dict[str, Any]] = None) -> ContextLoggerAdapter:
    """
    Get a logger with context support.
    
    Args:
        name: Logger name (typically __name__)
        context: Optional default context dict
        
    Returns:
        ContextLoggerAdapter instance
    """
    logger = logging.getLogger(name)
    return ContextLoggerAdapter(logger, context)


def set_request_context(request_id: str, operation: Optional[str] = None):
    """Set request context for logging"""
    request_id_context.set(request_id)
    if operation:
        operation_context.set(operation)


def clear_request_context():
    """Clear request context"""
    request_id_context.set(None)
    operation_context.set(None)


def log_function_call(func_name: str, **kwargs) -> Dict[str, Any]:
    """
    Create a context dict for logging function calls.
    
    Args:
        func_name: Function name
        **kwargs: Additional context fields
        
    Returns:
        Context dict for logging
    """
    return {
        "function": func_name,
        **kwargs
    }


def log_external_service_call(
    service: str,
    endpoint: str,
    method: str = "POST",
    **kwargs
) -> Dict[str, Any]:
    """
    Create a context dict for logging external service calls.
    
    Args:
        service: Service name (e.g., "llm", "embedding", "neo4j")
        endpoint: Endpoint/operation name
        method: HTTP method or operation type
        **kwargs: Additional context fields
        
    Returns:
        Context dict for logging
    """
    return {
        "service": service,
        "endpoint": endpoint,
        "method": method,
        **kwargs
    }


def log_database_operation(
    operation: str,
    query_type: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Create a context dict for logging database operations.
    
    Args:
        operation: Operation name
        query_type: Type of query (e.g., "cypher", "read", "write")
        **kwargs: Additional context fields
        
    Returns:
        Context dict for logging
    """
    return {
        "service": "neo4j",
        "operation": operation,
        "query_type": query_type,
        **kwargs
    }


def log_error_with_context(
    logger: logging.Logger,
    message: str,
    exception: Optional[Exception] = None,
    context: Optional[Dict[str, Any]] = None,
    level: str = "error"
):
    """
    Log an error with full context and exception information.
    
    Args:
        logger: Logger instance
        message: Error message
        exception: Optional exception object
        context: Additional context
        level: Log level ("error", "warning", "critical")
    """
    extra = context or {}
    if exception:
        extra["exception_type"] = type(exception).__name__
        extra["exception_message"] = str(exception)
    
    log_func = getattr(logger, level, logger.error)
    log_func(message, extra=extra, exc_info=exception is not None)


# Initialize logging on module import
def _init_logging():
    """Initialize logging when module is imported"""
    try:
        setup_logging()
    except Exception:
        # Fallback to basic logging if config not available
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )


# Initialize on import
_init_logging()

