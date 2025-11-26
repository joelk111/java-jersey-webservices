"""
LLM client for interacting with Ollama.
"""

import httpx
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from .config import config


@dataclass
class Message:
    """Represents a chat message."""
    role: str  # "system", "user", or "assistant"
    content: str


@dataclass
class ToolCall:
    """Represents a tool call from the LLM."""
    name: str
    arguments: Dict[str, Any]


class OllamaClient:
    """Client for interacting with Ollama LLM API."""

    def __init__(self):
        self.base_url = config.ollama_host
        self.model = config.ollama_model
        self.client = httpx.Client(timeout=config.llm_timeout)

    def chat(
        self,
        messages: List[Message],
        tools: Optional[List[Dict]] = None,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Send a chat request to Ollama with optional tool calling.

        Args:
            messages: List of Message objects
            tools: Optional list of tool definitions
            stream: Whether to stream the response

        Returns:
            Response dictionary from Ollama
        """
        payload = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": stream,
            "options": {
                "temperature": config.llm_temperature,
                "num_predict": config.llm_max_tokens,
            }
        }

        if tools:
            payload["tools"] = tools

        response = self.client.post(
            f"{self.base_url}/api/chat",
            json=payload
        )
        response.raise_for_status()
        return response.json()

    def generate(self, prompt: str, stream: bool = False) -> str:
        """
        Simple text generation without chat format.

        Args:
            prompt: The prompt to send
            stream: Whether to stream the response

        Returns:
            Generated text response
        """
        response = self.client.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": stream,
                "options": {
                    "temperature": config.llm_temperature,
                    "num_predict": config.llm_max_tokens,
                }
            }
        )
        response.raise_for_status()
        return response.json()["response"]

    def is_available(self) -> bool:
        """Check if Ollama is running and the model is available."""
        try:
            response = self.client.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                return any(m["name"].startswith(self.model.split(":")[0]) for m in models)
            return False
        except Exception:
            return False


# Tool definitions for Llama function calling
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "match_fields",
            "description": "Match user-mentioned field names to ORBIS data dictionary fields using fuzzy matching",
            "parameters": {
                "type": "object",
                "properties": {
                    "field_names": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of field names or descriptions mentioned by the user"
                    }
                },
                "required": ["field_names"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_rule",
            "description": "Generate a data quality validation rule in CSV format",
            "parameters": {
                "type": "object",
                "properties": {
                    "rule_type": {
                        "type": "string",
                        "enum": ["REGEX", "NOT_NULL", "LENGTH", "RANGE", "IN_LIST", "CUSTOM_FUNCTION"],
                        "description": "Type of validation rule"
                    },
                    "field_name": {
                        "type": "string",
                        "description": "Fully qualified field name (table.field)"
                    },
                    "validation_value": {
                        "type": "string",
                        "description": "Validation pattern, range, or function name"
                    },
                    "error_message": {
                        "type": "string",
                        "description": "Error message when validation fails"
                    },
                    "severity": {
                        "type": "string",
                        "enum": ["ERROR", "WARNING", "INFO"],
                        "description": "Severity level of the rule"
                    }
                },
                "required": ["rule_type", "field_name", "error_message"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_custom_function",
            "description": "Generate Python code for a custom validation function when standard rules are insufficient",
            "parameters": {
                "type": "object",
                "properties": {
                    "function_name": {
                        "type": "string",
                        "description": "Name of the validation function (snake_case)"
                    },
                    "description": {
                        "type": "string",
                        "description": "Detailed description of what the function should validate"
                    },
                    "parameters": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Additional parameters beyond the field value"
                    }
                },
                "required": ["function_name", "description"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "ask_clarification",
            "description": "Ask the user for clarification when information is missing or ambiguous",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The clarifying question to ask the user"
                    },
                    "options": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of choices to present"
                    }
                },
                "required": ["question"]
            }
        }
    }
]
