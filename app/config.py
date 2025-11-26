"""
Configuration management for the NLP Rules Engine.
"""

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv is optional


@dataclass
class Config:
    """Application configuration loaded from environment variables."""

    # Ollama settings
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.1:8b-instruct-q4_0")

    # Application settings
    app_host: str = os.getenv("APP_HOST", "0.0.0.0")
    app_port: int = int(os.getenv("APP_PORT", "7860"))
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"

    # Paths
    field_dictionary_path: str = os.getenv(
        "FIELD_DICTIONARY_PATH",
        str(Path(__file__).parent.parent / "orbis_field_names.c")
    )
    rules_output_dir: str = os.getenv(
        "RULES_OUTPUT_DIR",
        str(Path(__file__).parent.parent / "generated_rules")
    )

    # Fuzzy matching threshold (0-100)
    fuzzy_threshold: int = int(os.getenv("FUZZY_THRESHOLD", "70"))

    # LLM settings
    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.7"))
    llm_max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "2048"))
    llm_timeout: int = int(os.getenv("LLM_TIMEOUT", "120"))


# Global config instance
config = Config()
