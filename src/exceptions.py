
"""
exceptions.py - Custom exception hierarchy with automatic line, file, and function tracing.
Also includes trace_id propagation for correlation across asynchronous or distributed flows.
"""

import inspect
import sys
from typing import Optional, Dict, Any
from contextvars import ContextVar


# Trace ID context (can be set by logger or main entry point)
_trace_id_var: ContextVar[Optional[str]] = ContextVar("trace_id", default=None)

def set_trace_id(trace_id: str) -> None:
    """Set a trace ID for the current execution context (e.g., request ID)."""
    _trace_id_var.set(trace_id)

def get_trace_id() -> str:
    """Get current trace ID, or 'no-trace' if not set."""
    return _trace_id_var.get() or "no-trace"


# Helper to extract caller location (skip frames for internal methods)
def _get_caller_info(skip_frames: int = 2) -> Dict[str, Any]:
    """
    Returns a dictionary with 'file', 'line', 'function' of the caller.
    skip_frames: number of stack frames to go up (default 2 excludes this function and __init__).
    """
    try:
        # Get current frame
        frame = inspect.currentframe()
        # Move up the requested number of frames
        for _ in range(skip_frames):
            if frame is not None:
                frame = frame.f_back
            else:
                break
        if frame is not None:
            filename = frame.f_code.co_filename
            lineno = frame.f_lineno
            funcname = frame.f_code.co_name
            return {"file": filename, "line": lineno, "function": funcname}
    except Exception:
        # Fallback in case inspection fails
        pass
    return {"file": "unknown", "line": 0, "function": "unknown"}


# Base exception for all project-specific errors
class AppException(Exception):
    """
    Base exception for the entire application.
    Automatically captures the file, line, function, and current trace_id.
    """
    def __init__(self, message: str, *args, **kwargs):
        super().__init__(message, *args, **kwargs)
        self.message = message
        self.location = _get_caller_info(skip_frames=2)  # skip __init__ and this function
        self.trace_id = get_trace_id()

    def __str__(self) -> str:
        # Format: [file:line in function] [trace:xxx] message
        loc = f"[{self.location['file']}:{self.location['line']} in {self.location['function']}]"
        tid = f"[Trace: {self.trace_id}]"
        return f"{loc} {tid} {self.message}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging or API responses."""
        return {
            "type": self.__class__.__name__,
            "message": self.message,
            "file": self.location["file"],
            "line": self.location["line"],
            "function": self.location["function"],
            "trace_id": self.trace_id,
        }

"""Setting up standalone exceptions for component parts of the projrct pipeline"""


# Configuration & Environment Errors
class ConfigurationError(AppException):
    """Missing or invalid configuration (e.g., missing API keys, invalid paths)."""
    pass

class EnvironmentError(AppException):
    """Problems with the runtime environment (e.g., missing system dependencies)."""
    pass

# Document Parsing Errors
class ParsingError(AppException):
    """Base class for document parsing failures."""
    pass

class UnsupportedFileTypeError(ParsingError):
    """File extension not supported (.exe, .png, etc.)."""
    pass

class TextExtractionError(ParsingError):
    """Could not extract text from a supported file (corrupted, encrypted, empty)."""
    pass

class SectionNotFoundError(ParsingError):
    """Expected section (e.g., 'Experience') not found in document. Not always fatal."""
    pass


# LLM / Agent Errors
class AgentError(AppException):
    """Base class for agent-related errors."""
    pass

class LLMServiceError(AgentError):
    """Errors from LLM API calls (network, auth, rate limits, malformed responses)."""
    pass

class HallucinationDetectedError(AgentError):
    """Raised when an agent generated content not supported by source data (strict mode)."""
    pass

class ModelLoadError(AgentError):
    """Failed to load a local Hugging Face model (e.g., out of memory, missing weights)."""
    pass


# File I/O Errors
class FileOperationError(AppException):
    """Reading/writing files failed (permissions, disk full, etc.)."""
    pass

# LaTeX Compilation Errors
class LaTeXCompilationError(AppException):
    """LaTeX to PDF compilation failed (syntax errors, missing packages)."""
    pass