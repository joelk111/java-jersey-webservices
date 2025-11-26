# NLP Rules Engine - Part 3: Development Guide

## Project Structure

```
nlp-rules-engine/
├── app/
│   ├── __init__.py
│   ├── main.py              # Application entry point
│   ├── config.py            # Configuration management
│   ├── llm_client.py        # Ollama LLM interface
│   ├── field_matcher.py     # Fuzzy field matching
│   ├── rule_generator.py    # Rule creation logic
│   ├── code_generator.py    # Custom function generator
│   ├── conversation.py      # Multi-turn dialogue manager
│   └── ui.py                # Gradio web interface
├── tools/
│   ├── __init__.py
│   ├── match_fields.py      # Field matching tool
│   ├── generate_rule.py     # Rule generation tool
│   └── generate_code.py     # Code generation tool
├── templates/
│   ├── rule_csv.txt         # CSV rule template
│   ├── rule_json.txt        # JSON rule template
│   └── function.py.txt      # Python function template
├── tests/
│   ├── test_field_matcher.py
│   ├── test_rule_generator.py
│   └── test_llm_client.py
├── orbis_field_names.c      # Field dictionary
├── sample_rules.csv         # Example rules
├── rules.py                 # Custom functions
├── requirements.txt
├── .env
└── README.md
```

## Core Components

### 1. Configuration (config.py)

```python
# app/config.py
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Config:
    # Ollama settings
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.1:8b-instruct-q4_0")

    # Application settings
    app_host: str = os.getenv("APP_HOST", "0.0.0.0")
    app_port: int = int(os.getenv("APP_PORT", "7860"))
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"

    # Paths
    field_dictionary_path: str = os.getenv("FIELD_DICTIONARY_PATH", "./orbis_field_names.c")
    rules_output_dir: str = os.getenv("RULES_OUTPUT_DIR", "./generated_rules")

    # Fuzzy matching threshold
    fuzzy_threshold: int = int(os.getenv("FUZZY_THRESHOLD", "70"))

config = Config()
```

### 2. LLM Client (llm_client.py)

```python
# app/llm_client.py
import httpx
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from .config import config

@dataclass
class Message:
    role: str  # "system", "user", or "assistant"
    content: str

@dataclass
class ToolCall:
    name: str
    arguments: Dict[str, Any]

class OllamaClient:
    def __init__(self):
        self.base_url = config.ollama_host
        self.model = config.ollama_model
        self.client = httpx.Client(timeout=120.0)

    def chat(
        self,
        messages: List[Message],
        tools: Optional[List[Dict]] = None,
        stream: bool = False
    ) -> Dict[str, Any]:
        """Send chat request to Ollama with optional tool calling."""

        payload = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": stream,
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
        """Simple text generation."""
        response = self.client.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": stream
            }
        )
        response.raise_for_status()
        return response.json()["response"]

# Tool definitions for Llama
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "match_fields",
            "description": "Match user-mentioned field names to ORBIS data dictionary fields",
            "parameters": {
                "type": "object",
                "properties": {
                    "field_names": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of field names mentioned by the user"
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
            "description": "Generate a data quality rule in CSV format",
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
                        "description": "Validation pattern or value"
                    },
                    "error_message": {
                        "type": "string",
                        "description": "Error message when validation fails"
                    },
                    "severity": {
                        "type": "string",
                        "enum": ["ERROR", "WARNING", "INFO"],
                        "description": "Severity level"
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
            "description": "Generate Python code for a custom validation function",
            "parameters": {
                "type": "object",
                "properties": {
                    "function_name": {
                        "type": "string",
                        "description": "Name of the validation function"
                    },
                    "description": {
                        "type": "string",
                        "description": "What the function should validate"
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
            "description": "Ask user for clarification when information is missing",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The clarifying question to ask"
                    },
                    "options": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of choices"
                    }
                },
                "required": ["question"]
            }
        }
    }
]
```

### 3. Field Matcher (field_matcher.py)

```python
# app/field_matcher.py
import re
from typing import List, Tuple, Dict
from rapidfuzz import fuzz, process
from .config import config

class FieldMatcher:
    def __init__(self, field_dictionary_path: str = None):
        self.path = field_dictionary_path or config.field_dictionary_path
        self.fields: List[str] = []
        self.field_labels: Dict[str, str] = {}
        self.field_types: Dict[str, str] = {}
        self._load_fields()

    def _load_fields(self):
        """Load fields from the C-style header file."""
        with open(self.path, 'r') as f:
            content = f.read()

        # Parse lines like: "table.field",  // Label | TYPE
        pattern = r'"([^"]+)",\s*//\s*([^|]*)\|\s*(\w+)'

        for match in re.finditer(pattern, content):
            field_name = match.group(1)
            label = match.group(2).strip()
            data_type = match.group(3).strip()

            self.fields.append(field_name)
            self.field_labels[field_name] = label
            self.field_types[field_name] = data_type

    def match(self, query: str, limit: int = 5) -> List[Tuple[str, int, str]]:
        """
        Find best matching fields for a query string.

        Returns list of (field_name, score, label) tuples.
        """
        # Normalize query
        query = query.lower().replace(" ", "_").replace("-", "_")

        # Create searchable text for each field
        searchable = {}
        for field in self.fields:
            # Combine field name and label for matching
            label = self.field_labels.get(field, "")
            search_text = f"{field} {label}".lower()
            searchable[field] = search_text

        # Use rapidfuzz for fuzzy matching
        results = []
        for field, search_text in searchable.items():
            # Try multiple matching strategies
            scores = [
                fuzz.ratio(query, field.split('.')[-1]),  # Field name only
                fuzz.ratio(query, field),                  # Full field path
                fuzz.partial_ratio(query, search_text),    # Partial match with label
                fuzz.token_set_ratio(query, search_text),  # Token-based match
            ]
            best_score = max(scores)
            if best_score >= config.fuzzy_threshold:
                results.append((field, best_score, self.field_labels.get(field, "")))

        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]

    def match_multiple(self, queries: List[str]) -> Dict[str, List[Tuple[str, int, str]]]:
        """Match multiple field queries."""
        return {query: self.match(query) for query in queries}

    def get_field_info(self, field_name: str) -> Dict[str, str]:
        """Get information about a specific field."""
        if field_name in self.fields:
            return {
                "field_name": field_name,
                "label": self.field_labels.get(field_name, ""),
                "data_type": self.field_types.get(field_name, "STRING")
            }
        return None

    def search_by_table(self, table_name: str) -> List[str]:
        """Get all fields for a specific table."""
        return [f for f in self.fields if f.startswith(f"{table_name}.")]

    def list_tables(self) -> List[str]:
        """Get list of all table names."""
        tables = set()
        for field in self.fields:
            if '.' in field:
                tables.add(field.split('.')[0])
        return sorted(tables)
```

### 4. Rule Generator (rule_generator.py)

```python
# app/rule_generator.py
import csv
import json
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from .config import config

@dataclass
class Rule:
    rule_id: str
    rule_name: str
    rule_type: str
    field_name: str
    validation_type: str
    validation_value: str
    error_message: str
    severity: str
    is_active: bool = True

class RuleGenerator:
    def __init__(self):
        self.output_dir = config.rules_output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self._rule_counter = 0

    def _generate_rule_id(self) -> str:
        """Generate unique rule ID."""
        self._rule_counter += 1
        timestamp = datetime.now().strftime("%Y%m%d")
        return f"R{timestamp}_{self._rule_counter:04d}"

    def create_regex_rule(
        self,
        field_name: str,
        pattern: str,
        error_message: str,
        severity: str = "ERROR",
        rule_name: Optional[str] = None
    ) -> Rule:
        """Create a REGEX validation rule."""
        rule_id = self._generate_rule_id()
        if not rule_name:
            rule_name = f"Regex_Check_{field_name.replace('.', '_')}"

        return Rule(
            rule_id=rule_id,
            rule_name=rule_name,
            rule_type="REGEX",
            field_name=field_name,
            validation_type="REGEX",
            validation_value=pattern,
            error_message=error_message,
            severity=severity
        )

    def create_not_null_rule(
        self,
        field_name: str,
        error_message: str,
        severity: str = "ERROR"
    ) -> Rule:
        """Create a NOT_NULL validation rule."""
        rule_id = self._generate_rule_id()
        return Rule(
            rule_id=rule_id,
            rule_name=f"Required_{field_name.replace('.', '_')}",
            rule_type="NOT_NULL",
            field_name=field_name,
            validation_type="NOT_NULL",
            validation_value="",
            error_message=error_message,
            severity=severity
        )

    def create_range_rule(
        self,
        field_name: str,
        min_value: float,
        max_value: float,
        error_message: str,
        severity: str = "ERROR"
    ) -> Rule:
        """Create a RANGE validation rule."""
        rule_id = self._generate_rule_id()
        return Rule(
            rule_id=rule_id,
            rule_name=f"Range_Check_{field_name.replace('.', '_')}",
            rule_type="RANGE",
            field_name=field_name,
            validation_type="RANGE",
            validation_value=f"{min_value},{max_value}",
            error_message=error_message,
            severity=severity
        )

    def create_custom_function_rule(
        self,
        field_name: str,
        function_name: str,
        error_message: str,
        severity: str = "ERROR"
    ) -> Rule:
        """Create a CUSTOM_FUNCTION rule."""
        rule_id = self._generate_rule_id()
        return Rule(
            rule_id=rule_id,
            rule_name=f"Custom_{function_name}",
            rule_type="CUSTOM_FUNCTION",
            field_name=field_name,
            validation_type="CUSTOM_FUNCTION",
            validation_value=function_name,
            error_message=error_message,
            severity=severity
        )

    def to_csv_row(self, rule: Rule) -> str:
        """Convert rule to CSV row."""
        return ",".join([
            rule.rule_id,
            rule.rule_name,
            rule.rule_type,
            rule.field_name,
            rule.validation_type,
            f'"{rule.validation_value}"' if ',' in rule.validation_value else rule.validation_value,
            rule.error_message,
            rule.severity,
            "TRUE" if rule.is_active else "FALSE"
        ])

    def to_json(self, rule: Rule) -> Dict[str, Any]:
        """Convert rule to JSON format."""
        return {
            "rule_id": rule.rule_id,
            "rule_name": rule.rule_name,
            "type": rule.rule_type.lower(),
            "field": rule.field_name,
            "validation": {
                "type": rule.validation_type,
                "value": rule.validation_value
            },
            "error_message": rule.error_message,
            "severity": rule.severity,
            "is_active": rule.is_active
        }

    def save_rules(self, rules: list, filename: str = None) -> str:
        """Save rules to CSV file."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"rules_{timestamp}.csv"

        filepath = os.path.join(self.output_dir, filename)

        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            # Header
            writer.writerow([
                "rule_id", "rule_name", "rule_type", "field_name",
                "validation_type", "validation_value", "error_message",
                "severity", "is_active"
            ])
            # Data
            for rule in rules:
                writer.writerow([
                    rule.rule_id, rule.rule_name, rule.rule_type,
                    rule.field_name, rule.validation_type, rule.validation_value,
                    rule.error_message, rule.severity,
                    "TRUE" if rule.is_active else "FALSE"
                ])

        return filepath
```

### 5. Code Generator (code_generator.py)

```python
# app/code_generator.py
from typing import List, Optional
from dataclasses import dataclass
import os
from .config import config

@dataclass
class CustomFunction:
    name: str
    description: str
    parameters: List[str]
    code: str

class CodeGenerator:
    FUNCTION_TEMPLATE = '''
def {name}({params}) -> Tuple[bool, Optional[str]]:
    """
    {description}

    Args:
{args_doc}

    Returns:
        Tuple of (is_valid, error_message)
    """
{body}
'''

    def generate_function(
        self,
        name: str,
        description: str,
        validation_logic: str,
        parameters: Optional[List[str]] = None
    ) -> CustomFunction:
        """Generate a custom validation function."""

        params = ["value: str"]
        if parameters:
            params.extend(parameters)
        params_str = ", ".join(params)

        # Generate args documentation
        args_doc = "        value: The field value to validate"
        if parameters:
            for param in parameters:
                param_name = param.split(":")[0].strip()
                args_doc += f"\n        {param_name}: Additional parameter"

        # Generate function body
        body = f'''    if not value:
        return True, None  # Empty values handled by NOT_NULL rule

    {validation_logic}
'''

        code = self.FUNCTION_TEMPLATE.format(
            name=name,
            params=params_str,
            description=description,
            args_doc=args_doc,
            body=body
        )

        return CustomFunction(
            name=name,
            description=description,
            parameters=parameters or [],
            code=code
        )

    def generate_regex_function(
        self,
        name: str,
        pattern: str,
        error_message: str
    ) -> CustomFunction:
        """Generate a regex validation function."""

        validation_logic = f'''
    import re
    pattern = r"{pattern}"
    if not re.match(pattern, value):
        return False, "{error_message}"
    return True, None
'''

        return self.generate_function(
            name=name,
            description=f"Validates value against pattern: {pattern}",
            validation_logic=validation_logic
        )

    def generate_no_chars_function(
        self,
        name: str,
        disallowed_chars: str,
        error_message: str
    ) -> CustomFunction:
        """Generate function to check for disallowed characters."""

        validation_logic = f'''
    disallowed = "{disallowed_chars}"
    found = [c for c in disallowed if c in value]
    if found:
        return False, "{error_message}: " + ", ".join(found)
    return True, None
'''

        return self.generate_function(
            name=name,
            description=f"Checks that value does not contain: {disallowed_chars}",
            validation_logic=validation_logic
        )

    def save_function(self, func: CustomFunction, directory: str = None) -> str:
        """Save function to Python file."""
        if not directory:
            directory = os.path.join(os.path.dirname(config.rules_output_dir), "custom_functions")

        os.makedirs(directory, exist_ok=True)

        filepath = os.path.join(directory, f"{func.name}.py")

        header = '''"""
Auto-generated custom validation function.
"""
from typing import Tuple, Optional

'''

        with open(filepath, 'w') as f:
            f.write(header)
            f.write(func.code)

        return filepath
```

### 6. Conversation Manager (conversation.py)

```python
# app/conversation.py
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from .llm_client import OllamaClient, Message, TOOLS
from .field_matcher import FieldMatcher
from .rule_generator import RuleGenerator
from .code_generator import CodeGenerator

@dataclass
class ConversationState:
    messages: List[Message] = field(default_factory=list)
    pending_fields: List[str] = field(default_factory=list)
    matched_fields: Dict[str, str] = field(default_factory=dict)
    generated_rules: List[Any] = field(default_factory=list)
    generated_code: List[str] = field(default_factory=list)
    awaiting_clarification: bool = False
    clarification_context: Optional[Dict] = None

class ConversationManager:
    SYSTEM_PROMPT = """You are a data quality rules assistant. Help users create validation rules for the ORBIS data dictionary.

Your capabilities:
1. Match field names using fuzzy matching (call match_fields tool)
2. Generate validation rules in CSV format (call generate_rule tool)
3. Create custom Python validation functions (call generate_custom_function tool)
4. Ask clarifying questions when needed (call ask_clarification tool)

Available rule types:
- REGEX: Pattern matching validation
- NOT_NULL: Required field check
- LENGTH: String length validation
- RANGE: Numeric range validation
- IN_LIST: Allowed values list
- CUSTOM_FUNCTION: Custom Python function

When the user describes a rule:
1. First identify the field(s) they're referring to
2. Determine the appropriate rule type
3. If information is missing (like field name or severity), ask for clarification
4. Generate the rule and any required custom code

Always output rules in CSV format with columns:
rule_id, rule_name, rule_type, field_name, validation_type, validation_value, error_message, severity, is_active
"""

    def __init__(self):
        self.llm = OllamaClient()
        self.field_matcher = FieldMatcher()
        self.rule_generator = RuleGenerator()
        self.code_generator = CodeGenerator()
        self.state = ConversationState()

        # Initialize with system message
        self.state.messages.append(Message(
            role="system",
            content=self.SYSTEM_PROMPT
        ))

    def process_message(self, user_input: str) -> Dict[str, Any]:
        """Process user message and return response."""

        # Add user message
        self.state.messages.append(Message(role="user", content=user_input))

        # Call LLM with tools
        response = self.llm.chat(
            messages=self.state.messages,
            tools=TOOLS
        )

        # Process response
        result = self._process_response(response)

        # Add assistant message
        self.state.messages.append(Message(
            role="assistant",
            content=result.get("message", "")
        ))

        return result

    def _process_response(self, response: Dict) -> Dict[str, Any]:
        """Process LLM response and execute any tool calls."""

        message = response.get("message", {})
        content = message.get("content", "")
        tool_calls = message.get("tool_calls", [])

        result = {
            "message": content,
            "rules": [],
            "code": [],
            "clarification": None,
            "matched_fields": {}
        }

        for tool_call in tool_calls:
            tool_name = tool_call.get("function", {}).get("name")
            arguments = tool_call.get("function", {}).get("arguments", {})

            if tool_name == "match_fields":
                matches = self._execute_match_fields(arguments)
                result["matched_fields"] = matches

            elif tool_name == "generate_rule":
                rule = self._execute_generate_rule(arguments)
                result["rules"].append(rule)

            elif tool_name == "generate_custom_function":
                code = self._execute_generate_code(arguments)
                result["code"].append(code)

            elif tool_name == "ask_clarification":
                result["clarification"] = arguments
                self.state.awaiting_clarification = True
                self.state.clarification_context = arguments

        return result

    def _execute_match_fields(self, args: Dict) -> Dict[str, List]:
        """Execute field matching."""
        field_names = args.get("field_names", [])
        matches = self.field_matcher.match_multiple(field_names)

        # Store matched fields
        for query, results in matches.items():
            if results:
                self.state.matched_fields[query] = results[0][0]  # Best match

        return matches

    def _execute_generate_rule(self, args: Dict) -> Dict:
        """Execute rule generation."""
        rule_type = args.get("rule_type", "REGEX")
        field_name = args.get("field_name", "")
        validation_value = args.get("validation_value", "")
        error_message = args.get("error_message", "Validation failed")
        severity = args.get("severity", "ERROR")

        if rule_type == "REGEX":
            rule = self.rule_generator.create_regex_rule(
                field_name=field_name,
                pattern=validation_value,
                error_message=error_message,
                severity=severity
            )
        elif rule_type == "NOT_NULL":
            rule = self.rule_generator.create_not_null_rule(
                field_name=field_name,
                error_message=error_message,
                severity=severity
            )
        elif rule_type == "CUSTOM_FUNCTION":
            rule = self.rule_generator.create_custom_function_rule(
                field_name=field_name,
                function_name=validation_value,
                error_message=error_message,
                severity=severity
            )
        else:
            rule = self.rule_generator.create_regex_rule(
                field_name=field_name,
                pattern=validation_value,
                error_message=error_message,
                severity=severity
            )

        self.state.generated_rules.append(rule)

        return self.rule_generator.to_json(rule)

    def _execute_generate_code(self, args: Dict) -> str:
        """Execute code generation."""
        func_name = args.get("function_name", "custom_validation")
        description = args.get("description", "Custom validation")
        params = args.get("parameters", [])

        # Determine validation type from description
        if "exclamation" in description.lower() or "special char" in description.lower():
            func = self.code_generator.generate_no_chars_function(
                name=func_name,
                disallowed_chars="!",
                error_message="Value contains disallowed characters"
            )
        else:
            func = self.code_generator.generate_function(
                name=func_name,
                description=description,
                validation_logic="# TODO: Implement validation logic\n    return True, None",
                parameters=params
            )

        self.state.generated_code.append(func.code)

        return func.code

    def get_rules_csv(self) -> str:
        """Get all generated rules as CSV."""
        if not self.state.generated_rules:
            return ""

        filepath = self.rule_generator.save_rules(self.state.generated_rules)
        with open(filepath, 'r') as f:
            return f.read()

    def reset(self):
        """Reset conversation state."""
        self.state = ConversationState()
        self.state.messages.append(Message(
            role="system",
            content=self.SYSTEM_PROMPT
        ))
```

## Running the Application

See `app/main.py` for the complete application entry point and Gradio UI setup.

## Next Steps

Proceed to **Part 4: Testing and QA** for:
- Unit test examples
- Integration test scenarios
- Manual QA checklist
