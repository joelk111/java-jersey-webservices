"""
Environment-based configuration management for the NLP Rules Engine.

Supports multiple environments: development, staging, production
Configuration is loaded from environment variables or .env files.
"""

import os
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

# Try to load dotenv
try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


def load_environment_config(env_name: Optional[str] = None) -> None:
    """
    Load configuration from environment-specific .env file.

    Args:
        env_name: Environment name (development, staging, production)
                  If not provided, uses ENVIRONMENT env var or defaults to development
    """
    if load_dotenv is None:
        return

    if env_name is None:
        env_name = os.getenv("ENVIRONMENT", "development")

    project_root = get_project_root()

    # Try environment-specific config first
    env_file = project_root / "config" / f"{env_name}.env"
    if env_file.exists():
        load_dotenv(env_file)
        return

    # Fall back to root .env
    root_env = project_root / ".env"
    if root_env.exists():
        load_dotenv(root_env)


# Load environment config on import
load_environment_config()


@dataclass
class Config:
    """Application configuration loaded from environment variables."""

    # Environment
    environment: str = os.getenv("ENVIRONMENT", "development")

    # Ollama settings
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.1:8b-instruct-q4_0")

    # Application settings
    app_host: str = os.getenv("APP_HOST", "0.0.0.0")
    app_port: int = int(os.getenv("APP_PORT", "7860"))
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    # Paths (resolved relative to project root)
    field_dictionary_path: str = os.getenv(
        "FIELD_DICTIONARY_PATH",
        str(get_project_root() / "orbis_field_names.c")
    )
    rules_output_dir: str = os.getenv(
        "RULES_OUTPUT_DIR",
        str(get_project_root() / "generated_rules")
    )
    custom_functions_dir: str = os.getenv(
        "CUSTOM_FUNCTIONS_DIR",
        str(get_project_root() / "custom_functions")
    )

    # Fuzzy matching threshold (0-100)
    fuzzy_threshold: int = int(os.getenv("FUZZY_THRESHOLD", "70"))

    # LLM settings
    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.7"))
    llm_max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "2048"))
    llm_timeout: int = int(os.getenv("LLM_TIMEOUT", "120"))

    # UI settings
    ui_theme: str = os.getenv("UI_THEME", "soft")
    ui_share: bool = os.getenv("UI_SHARE", "false").lower() == "true"

    # Security (optional)
    gradio_auth: Optional[Tuple[str, str]] = field(default=None)

    def __post_init__(self):
        """Post-initialization setup."""
        # Setup authentication if credentials provided
        username = os.getenv("GRADIO_AUTH_USERNAME")
        password = os.getenv("GRADIO_AUTH_PASSWORD")
        if username and password:
            self.gradio_auth = (username, password)

        # Ensure output directories exist
        Path(self.rules_output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.custom_functions_dir).mkdir(parents=True, exist_ok=True)

        # Setup logging
        logging.basicConfig(
            level=getattr(logging, self.log_level.upper(), logging.INFO),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == "production"

    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == "development"


# Global config instance
config = Config()

# Export logger
logger = logging.getLogger("nlp_rules_engine")
