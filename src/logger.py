
"""
logger.py - Centralized logging setup for the entire project.
Provides a get_logger() function that returns a configured logger with:
- Trace ID injection (reads from exceptions module's trace context)
- Colored console output using colorlog (if available)
- Optional JSON logging for production
- Consistent format across all modules
"""

import logging
import sys
import os
from typing import Optional


# Import trace_id from exceptions module (lazy to avoid circular imports)
def _get_trace_id() -> str:
    try:
        from exceptions import get_trace_id
        return get_trace_id()
    except ImportError:
        return "no-trace-id"


# Filter to inject trace_id into every log record

class TraceIdFilter(logging.Filter):
    """Add trace_id attribute to every log record."""
    def filter(self, record: logging.LogRecord) -> bool:
        record.trace_id = _get_trace_id()
        return True

# Load log level from environment

def _get_log_level() -> int:
    level_str = os.getenv("LOG_LEVEL", "INFO").upper()
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    return level_map.get(level_str, logging.INFO)


# Formatters (plain and JSON)
class PlainFormatter(logging.Formatter):
    """Plain text formatter with trace_id."""
    def format(self, record: logging.LogRecord) -> str:
        # Ensure trace_id is present (filter already does, but fallback)
        if not hasattr(record, 'trace_id'):
            record.trace_id = _get_trace_id()
        return super().format(record)

class JSONFormatter(PlainFormatter):
    """JSON formatter for structured logging."""
    def format(self, record: logging.LogRecord) -> str:
        import json
        # Ensure trace_id
        if not hasattr(record, 'trace_id'):
            record.trace_id = _get_trace_id()
        log_entry = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "name": record.name,
            "level": record.levelname,
            "trace_id": record.trace_id,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry)


# Colored formatter using colorlog (with fallback)
class ColoredFormatter(PlainFormatter):
    """Colorful console formatter using colorlog if available."""
    def __init__(self, fmt: Optional[str] = None):
        if fmt is None:
            fmt = "%(asctime)s - %(name)s - %(levelname)s - [Trace: %(trace_id)s] - %(message)s"
        super().__init__(fmt)
        try:
            import colorlog
            self.color_formatter = colorlog.ColoredFormatter(
                fmt,
                datefmt="%Y-%m-%d %H:%M:%S",
                reset=True,
                log_colors={
                    'DEBUG': 'cyan',
                    'INFO': 'green',
                    'WARNING': 'yellow',
                    'ERROR': 'red',
                    'CRITICAL': 'red,bg_white',
                }
            )
            self.use_color = True
        except ImportError:
            self.use_color = False

    def format(self, record: logging.LogRecord) -> str:
        # Ensure trace_id
        if not hasattr(record, 'trace_id'):
            record.trace_id = _get_trace_id()
        if self.use_color:
            return self.color_formatter.format(record)
        else:
            return super().format(record)


# Logger initialization (singleton)
_initialized = False

def setup_logging(
    level: Optional[str] = None,
    json_output: bool = False,
    log_file: Optional[str] = None,
) -> None:
    """Configure the root logger once."""
    global _initialized
    if _initialized:
        return

    # Set log level
    log_level = _get_log_level() if level is None else getattr(logging, level.upper())

    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Add trace_id filter to root logger
    root_logger.addFilter(TraceIdFilter())

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    if json_output:
        console_handler.setFormatter(JSONFormatter())
    else:
        fmt = "%(asctime)s - %(name)s - %(levelname)s - [Trace: %(trace_id)s] - %(message)s"
        console_handler.setFormatter(ColoredFormatter(fmt))
    root_logger.addHandler(console_handler)

    # Optional file handler
    if log_file:
        try:
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            if json_output:
                file_handler.setFormatter(JSONFormatter())
            else:
                file_handler.setFormatter(PlainFormatter("%(asctime)s - %(name)s - %(levelname)s - [Trace: %(trace_id)s] - %(message)s"))
            root_logger.addHandler(file_handler)
        except Exception as e:
            root_logger.warning(f"Could not set up file logging to {log_file}: {e}")

    _initialized = True

def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a configured logger instance."""
    if not _initialized:
        setup_logging()
    if name is None:
        return logging.getLogger()
    return logging.getLogger(name)

def set_trace_id(trace_id: str) -> None:
    """Set trace ID in the exceptions module context."""
    try:
        from exceptions import set_trace_id as set_exception_trace_id
        set_exception_trace_id(trace_id)
    except ImportError:
        pass


# CLI test
if __name__ == "__main__":
    setup_logging(level="DEBUG", json_output=False)
    logger = get_logger("test")
    set_trace_id("test-456")
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning")
    try:
        raise ValueError("Example error")
    except ValueError as e:
        logger.error("An error occurred", exc_info=True)