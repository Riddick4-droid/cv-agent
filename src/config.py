
"""
config.py - Central configuration loader with validation and type safety.
Loads environment variables from .env file, validates required fields,
and provides a Config dataclass (using Pydantic) for easy access.
Also creates required directories automatically.
"""

import os
from pathlib import Path
from typing import Optional, Literal
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator, ValidationError

from src.exceptions import ConfigurationError
from src.logger import get_logger

logger = get_logger(__name__)


# Find project root and load .env
def _find_project_root() -> Path:
    """
    Find project root by looking for .env file in current directory or parents.
    Returns the directory containing .env, or current directory if not found.
    """
    current = Path(__file__).parent.resolve()
    for parent in [current] + list(current.parents):
        if (parent / ".env").exists():
            return parent
    return current

PROJECT_ROOT = _find_project_root() #this returns the dir that contains the  .env file 
ENV_PATH = PROJECT_ROOT / ".env"

if ENV_PATH.exists():
    load_dotenv(ENV_PATH)
    logger.debug(f"Loaded environment from {ENV_PATH}")
else:
    logger.warning(f"No .env file found at {ENV_PATH}. Using system environment variables.")


# Pydantic configuration model
class AppConfig(BaseModel):
    """
    Application configuration with validation and defaults.
    All fields are read from environment variables.
    """
    # LLM Provider 
    openai_api_key: str = Field(default="", description="OpenAI API key")
    use_local_fallback: bool = Field(default=False, description="Enable local DeepSeek fallback")
    local_model_path: str = Field(default="deepseek-ai/deepseek-llm-7b-chat", description="HuggingFace model ID or path")
    quantization_bits: Literal[4, 8] = Field(default=4, description="Quantization bits (4 or 8)")
    device: str = Field(default="auto", description="Device: auto, cuda, cpu")
    
    # LLM Request Settings where ge-greater than or equal to and le means less than or equal to
    llm_model: str = Field(default="gpt-4.1 mini", description="OpenAI model name")
    llm_temperature: float = Field(default=0.2, ge=0.0, le=2.0, description="Sampling temperature")
    llm_max_tokens: int = Field(default=5000, ge=1, le=100000, description="Max tokens per request")
    llm_max_retries: int = Field(default=3, ge=0, le=10, description="Number of retries on API failure")
    llm_timeout: int = Field(default=60, ge=10, description="Timeout in seconds")
    
    # Directory Paths
    data_dir: Path = Field(default=Path("./data"), description="Data directory for inputs")
    outputs_dir: Path = Field(default=Path("./outputs"), description="Outputs directory")
    templates_dir: Path = Field(default=Path("./templates"), description="Templates directory")
    logs_dir: Path = Field(default=Path("./logs"), description="Logs directory")
    
    #Logging 
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(default="INFO")
    log_json: bool = Field(default=False, description="JSON formatted logs")
    log_file: Optional[Path] = Field(default=None, description="Optional log file path")
    
    # Tracing
    trace_enabled: bool = Field(default=True, description="Enable trace ID injection")
    
    #LaTeX
    latex_compiler: Literal["auto", "tectonic", "pdflatex", "latexmk", "none"] = Field(default="auto")
    latex_cleanup: bool = Field(default=True, description="Clean temporary files after compilation")
    
    #Application Mode
    app_mode: Literal["development", "testing", "production"] = Field(default="development")
    
    # Validators
    @field_validator("openai_api_key", mode="before")
    @classmethod
    def validate_openai_key(cls, v: str) -> str: #checks for the openai api key
        """Warn if OpenAI key is missing but fallback is disabled."""
        if not v or v == "your-openai-api-key-here":
            # Allow empty, but will raise error only when used and fallback false
            return ""
        return v
    
    @field_validator("data_dir", "outputs_dir", "templates_dir", "logs_dir", mode="after")
    @classmethod
    def convert_paths(cls, v: Path) -> Path: 
        """Convert string paths to Path objects and resolve relative paths."""
        if isinstance(v, str):
            v = Path(v)
        # Resolve relative paths against project root
        if not v.is_absolute():
            v = PROJECT_ROOT / v
        return v
    
    # Model configuration (extract from environment)
    @classmethod
    def from_env(cls) -> "AppConfig":
        """Create config from environment variables with proper type conversion."""
        config_dict = {}
        
        # Helper to get env var with fallback
        def get_env(key: str, default=None, type_func=str):
            val = os.getenv(key, default)
            if val is None:
                return default
            try:
                return type_func(val)
            except (ValueError, TypeError):
                return default
        
        # Map environment variables to config fields
        config_dict["openai_api_key"] = get_env("OPENAI_API_KEY", "")
        config_dict["use_local_fallback"] = get_env("USE_LOCAL_FALLBACK", "true").lower() == "true"
        config_dict["local_model_path"] = get_env("LOCAL_MODEL_PATH", "deepseek-ai/deepseek-llm-7b-chat")
        config_dict["quantization_bits"] = get_env("QUANTIZATION_BITS", 4, int)
        config_dict["device"] = get_env("DEVICE", "auto")
        config_dict["llm_model"] = get_env("LLM_MODEL", "gpt-4.1 mini")
        config_dict["llm_temperature"] = get_env("LLM_TEMPERATURE", 0.2, float)
        config_dict["llm_max_tokens"] = get_env("LLM_MAX_TOKENS", 2000, int)
        config_dict["llm_max_retries"] = get_env("LLM_MAX_RETRIES", 3, int)
        config_dict["llm_timeout"] = get_env("LLM_TIMEOUT", 60, int)
        config_dict["data_dir"] = get_env("DATA_DIR", "./data")
        config_dict["outputs_dir"] = get_env("OUTPUTS_DIR", "./outputs")
        config_dict["templates_dir"] = get_env("TEMPLATES_DIR", "./templates")
        config_dict["logs_dir"] = get_env("LOGS_DIR", "./logs")
        config_dict["log_level"] = get_env("LOG_LEVEL", "INFO")
        config_dict["log_json"] = get_env("LOG_JSON", "false").lower() == "true"
        config_dict["log_file"] = get_env("LOG_FILE", None)
        config_dict["trace_enabled"] = get_env("TRACE_ENABLED", "true").lower() == "true"
        config_dict["latex_compiler"] = get_env("LATEX_COMPILER", "auto")
        config_dict["latex_cleanup"] = get_env("LATEX_CLEANUP", "true").lower() == "true"
        config_dict["app_mode"] = get_env("APP_MODE", "development")
        
        return cls(**config_dict)
    
    # Validation method (called after creation)
    def validate_critical(self) -> None:
        """
        Check critical configuration.
        Raises ConfigurationError if OpenAI key missing and fallback disabled.
        """
        if not self.openai_api_key and not self.use_local_fallback:
            raise ConfigurationError(
                "OPENAI_API_KEY is not set and USE_LOCAL_FALLBACK=false. "
                "Either provide an OpenAI key or enable local fallback."
            )
        if self.quantization_bits not in (4, 8):
            raise ConfigurationError(f"QUANTIZATION_BITS must be 4 or 8, got {self.quantization_bits}")
        logger.info("Configuration validation passed")
    
    # Directory creation
    def create_directories(self) -> None:
        """Create all required directories if they don't exist."""
        dirs = [self.data_dir, self.outputs_dir, self.templates_dir, self.logs_dir]
        for d in dirs:
            if d and not d.exists():
                d.mkdir(parents=True, exist_ok=True)
                logger.debug(f"Created directory: {d}")
    
    def __repr__(self) -> str:
        # Redact API key in repr for safety
        api_key_redacted = "***" if self.openai_api_key else "missing"
        return f"AppConfig(openai_api_key={api_key_redacted}, llm_model={self.llm_model}, app_mode={self.app_mode})"

# Global config instance
def load_config() -> AppConfig:
    """Load, validate, and prepare configuration."""
    config = AppConfig.from_env()
    config.create_directories()
    config.validate_critical()
    return config

# Singleton instance (loaded once)
_config = None

def get_config() -> AppConfig:
    """Return the global config instance, loading it if necessary."""
    global _config
    if _config is None:
        _config = load_config()
    return _config

# For backward compatibility (direct import). optiobnal
config = get_config()

# CLI test
if __name__ == "__main__":
    try:
        cfg = get_config()
        print(f"Configuration loaded successfully:")
        print(f"  Project root: {PROJECT_ROOT}")
        print(f"  OpenAI API key: {'***' if cfg.openai_api_key else 'missing'}")
        print(f"  Use local fallback: {cfg.use_local_fallback}")
        print(f"  Local model: {cfg.local_model_path}")
        print(f"  Quantization bits: {cfg.quantization_bits}")
        print(f"  LLM model: {cfg.llm_model}")
        print(f"  Directories: data={cfg.data_dir}, outputs={cfg.outputs_dir}, logs={cfg.logs_dir}")
        print(f"  App mode: {cfg.app_mode}")
    except ConfigurationError as e:
        print(f"Configuration error: {e}")
    except ValidationError as e:
        print(f"Validation error: {e}")