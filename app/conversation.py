"""
Enhanced conversation management for multi-turn dialogue with the LLM.

This module handles:
- Multi-turn conversation state
- Required field detection and clarification requests
- Rule generation in CSV and JSON formats
- Custom function generation
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import json
import re
import logging

from .llm_client import OllamaClient, Message, TOOLS
from .field_matcher import FieldMatcher
from .rule_generator import RuleGenerator
from .code_generator import CodeGenerator
from .requirements_validator import RequirementsValidator, RuleType
from .config import config

# Setup logger
logger = logging.getLogger("nlp_rules_engine")


@dataclass
class ConversationState:
    """Tracks the state of the conversation."""
    messages: List[Message] = field(default_factory=list)
    pending_fields: List[str] = field(default_factory=list)
    matched_fields: Dict[str, str] = field(default_factory=dict)
    generated_rules: List[Any] = field(default_factory=list)
    generated_json_rules: List[Dict] = field(default_factory=list)
    generated_code: List[str] = field(default_factory=list)
    awaiting_clarification: bool = False
    clarification_context: Optional[Dict] = None
    last_rule_type: Optional[RuleType] = None
    extracted_info: Dict[str, str] = field(default_factory=dict)


class ConversationManager:
    """Manages multi-turn conversations for rule creation with required field detection."""

    SYSTEM_PROMPT = """You are a data quality rules assistant for the ORBIS data dictionary. Your job is to help users create validation rules using natural language AND answer questions about rules, validation types, and how the system works.

## YOUR CAPABILITIES:

1. **Create data quality rules** - Generate CSV and JSON rules from natural language
2. **Answer questions** - Explain rule types, show examples, describe custom functions
3. **Provide guidance** - Help users understand what validations are available

## RULE TYPES AND EXAMPLES:

### 1. REGEX (Pattern Matching)
Example rules:
- R001: BVD ID must start with 2-letter country code: `^[A-Z]{2}[A-Z0-9]+$`
- R002: Email format validation: `^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$`
- R003: URL should not contain exclamation points: `^[^!]*$`
- R006: Phone number format: `^\+?[0-9\s\-\(\)]+$`
- R007: Date format YYYY-MM-DD: `^\d{4}-\d{2}-\d{2}$`
- R011: URL must start with http/https: `^https?://`
REQUIRED: field_name, pattern

### 2. NOT_NULL (Required Field)
Example: R005: Postcode is required
REQUIRED: field_name only

### 3. LENGTH (String Length)
Example: R004: Country ISO code must be exactly 2 characters (min=2, max=2)
REQUIRED: field_name, min_length, max_length

### 4. RANGE (Numeric Range)
Examples:
- R008: Number of employees must be between 0 and 10,000,000
- R009: Percentage must be between 0 and 100
- R013: Revenue must be positive (0 to 999999999999)
- R014: Year must be between 1800 and 2025
REQUIRED: field_name, min_value, max_value

### 5. IN_LIST (Allowed Values)
Example: R015: Registration status must be one of: Active, Inactive, Pending, Dissolved, Merged
REQUIRED: field_name, allowed_values

### 6. CUSTOM_FUNCTION (Python Functions)
Example: R010: BVD ID checksum validation using validate_bvd_checksum
Available custom functions:
- validate_bvd_checksum: Validates BvD ID format and checksum
- validate_url_no_special_chars: Checks URL for disallowed characters
- validate_corporate_structure: Validates ownership hierarchy (circular refs, max depth)
- validate_date_sequence: Ensures date1 < date2
- validate_percentage_sum: Checks percentages sum to 100%
- validate_email_domain: Validates against allowed/blocked domain lists
REQUIRED: field_name, function_name

## JSON RULES (Complex Validations):

### Composite (Multiple Conditions with AND/OR)
Example: "Address requires city AND country_iso_code AND postcode"
```json
{"type": "composite", "operator": "AND", "rules": [
  {"field": "city", "validation": "NOT_NULL"},
  {"field": "country_iso_code", "validation": "REGEX", "pattern": "^[A-Z]{2}$"},
  {"field": "postcode", "validation": "NOT_NULL"}
]}
```

### Conditional (If-Then)
Example: "If has_email is Y, then email_address must be valid"
```json
{"type": "conditional",
 "if": {"field": "has_email", "equals": "Y"},
 "then": {"field": "email_address", "validation": "REGEX", "pattern": "..."}}
```

### Cross-Field (Compare Two Fields)
Example: "Incorporation date must be before dissolution date"
```json
{"type": "cross_field", "validation": "LESS_THAN",
 "field1": "incorporation_date", "field2": "dissolution_date"}
```

## CRITICAL: Required Information Detection

Before generating ANY rule, you MUST have ALL required information. If MISSING, ask using ask_clarification tool.

## When User Asks Questions:

If user asks about:
- "What rule types are supported?" → List and explain the 6 types above
- "Show me an example of..." → Provide relevant example from the list above
- "What custom functions are available?" → List the custom functions
- "How do I create a composite rule?" → Explain JSON composite format
- "Can I compare two fields?" → Explain cross-field validation

## When User Wants to Create a Rule:

1. ANALYZE the request
2. IDENTIFY which rule type fits
3. CHECK if all required fields are provided
4. If MISSING: Call ask_clarification with specific question
5. If COMPLETE: Call match_fields then generate_rule

## Examples of When to Ask Clarification:

User: "Create a rule that checks if a field is NULL"
→ ask_clarification: "Which field should be required?"

User: "Validate the email format"
→ ask_clarification: "Which email field? (e.g., email_address)"

## Examples of Complete Requests (No Clarification Needed):

User: "Create a rule for web_url that checks for exclamation points"
→ Generate REGEX rule with field=web_url, pattern=^[^!]*$

User: "Make sure email_address is not empty"
→ Generate NOT_NULL rule with field=email_address

## Handling Typos and Informal Language:

- Typos: "cehck" = "check", "ruel" = "rule"
- Informal: "make sure", "gotta have", "can't be empty"
- Variations: "exclamation point", "!", "bang"

## DEVELOPER GUIDE KNOWLEDGE:

### Adding Custom Validation Functions

Custom functions go in rules.py with this signature:
```python
def validate_my_function(value: str, param1: str = None) -> Tuple[bool, Optional[str]]:
    if not value:
        return True, None  # Empty passes, use NOT_NULL for required
    # Validation logic
    if some_condition(value):
        return True, None
    return False, "Error message"

# Register in CUSTOM_FUNCTIONS dict
CUSTOM_FUNCTIONS['validate_my_function'] = validate_my_function
```

Then use in CSV rules:
```csv
R001,My_Rule,CUSTOM_FUNCTION,my_field,CUSTOM_FUNCTION,validate_my_function,Validation failed,ERROR,TRUE
```

### Field Matching API

The FieldMatcher searches 5,095+ ORBIS fields across 109 tables:
- `matcher.match("web_url", limit=5)` - Returns [(field_name, score, label), ...]
- `matcher.match_multiple(["email", "url"])` - Match multiple fields at once
- `matcher.list_tables()` - List all available tables

### Rule Generator API

- `generator.create_regex_rule(field_name, pattern, error_message, severity)`
- `generator.create_not_null_rule(field_name, error_message)`
- `generator.create_length_rule(field_name, min_length, max_length, error_message)`
- `generator.create_range_rule(field_name, min_value, max_value, error_message)`
- `generator.create_in_list_rule(field_name, allowed_values, error_message)`
- `generator.create_custom_function_rule(field_name, function_name, error_message)`

### CSV Rule Format

```
rule_id,rule_name,rule_type,field_name,validation_type,validation_value,error_message,severity,is_active
```

Severities: ERROR, WARNING
is_active: TRUE, FALSE

### Environment Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| OLLAMA_HOST | Ollama API URL | http://localhost:11434 |
| OLLAMA_MODEL | LLM model | llama3.1:8b-instruct-q4_0 |
| FUZZY_THRESHOLD | Field match threshold | 70 |
| LLM_TEMPERATURE | LLM creativity | 0.7 |

Be helpful and conversational while ensuring all requirements are met before generating rules."""

    def __init__(self):
        """Initialize the conversation manager."""
        self.llm = OllamaClient()
        self.field_matcher = FieldMatcher()
        self.rule_generator = RuleGenerator()
        self.code_generator = CodeGenerator()
        self.validator = RequirementsValidator()
        self.state = ConversationState()

        # Initialize with system message
        self.state.messages.append(Message(
            role="system",
            content=self.SYSTEM_PROMPT
        ))

        logger.info("ConversationManager initialized")

    def process_message(self, user_input: str) -> Dict[str, Any]:
        """
        Process a user message and return the response.

        Args:
            user_input: The user's message

        Returns:
            Dictionary with 'message', 'rules', 'json_rules', 'code',
            'clarification', 'matched_fields'
        """
        logger.debug(f"Processing user input: {user_input}")

        # Add user message to history
        self.state.messages.append(Message(role="user", content=user_input))

        # Pre-validate to catch obvious missing required fields
        pre_validation = self._pre_validate_request(user_input)

        if not pre_validation["is_valid"] and pre_validation["clarification"]:
            # Missing required information - ask for clarification
            result = {
                "message": pre_validation["clarification"],
                "rules": [],
                "json_rules": [],
                "code": [],
                "clarification": {"question": pre_validation["clarification"]},
                "matched_fields": pre_validation.get("matched_fields", {})
            }

            self.state.awaiting_clarification = True
            self.state.clarification_context = pre_validation
            self.state.extracted_info.update(pre_validation.get("extracted", {}))

            self.state.messages.append(Message(
                role="assistant",
                content=result["message"]
            ))

            return result

        # Proceed with LLM processing
        try:
            response = self.llm.chat(
                messages=self.state.messages,
                tools=TOOLS
            )
            result = self._process_response(response)

        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            result = self._fallback_processing(user_input, str(e))

        # Add assistant response to history
        if result.get("message"):
            self.state.messages.append(Message(
                role="assistant",
                content=result["message"]
            ))

        return result

    def _pre_validate_request(self, user_input: str) -> Dict[str, Any]:
        """
        Pre-validate request to catch missing required information.
        """
        input_lower = user_input.lower()

        # If awaiting clarification, user is providing missing info
        if self.state.awaiting_clarification:
            self.state.awaiting_clarification = False
            return {"is_valid": True, "clarification": None}

        # Extract field names mentioned
        extracted = {}
        matched_fields = {}

        potential_fields = re.findall(r'\b([a-z][a-z0-9_]*(?:_[a-z0-9]+)+)\b', input_lower)
        for pf in potential_fields:
            matches = self.field_matcher.match(pf, limit=1)
            if matches:
                matched_fields[pf] = matches[0]
                extracted["field_name"] = matches[0][0]

        # Detect rule type
        validation_result = self.validator.validate_request(user_input, extracted)

        # Check if field_name is missing
        if "field_name" in validation_result.missing_required:
            field_words = re.findall(r'\b(field|column|url|email|address|name|id|date|number|code)\b', input_lower)

            if not potential_fields and not field_words:
                return {
                    "is_valid": False,
                    "clarification": self.validator.format_missing_fields_message(validation_result),
                    "matched_fields": matched_fields,
                    "extracted": extracted,
                    "rule_type": validation_result.detected_rule_type
                }

        # Allow LLM to handle borderline cases
        if not validation_result.is_valid and len(validation_result.missing_required) <= 1:
            # LLM can often infer single missing fields
            return {"is_valid": True, "clarification": None, "matched_fields": matched_fields}

        return {
            "is_valid": validation_result.is_valid,
            "clarification": None if validation_result.is_valid else self.validator.format_missing_fields_message(validation_result),
            "matched_fields": matched_fields,
            "extracted": extracted,
            "rule_type": validation_result.detected_rule_type
        }

    def _process_response(self, response: Dict) -> Dict[str, Any]:
        """Process the LLM response and execute tool calls."""
        message = response.get("message", {})
        content = message.get("content", "")
        tool_calls = message.get("tool_calls", [])

        result = {
            "message": content,
            "rules": [],
            "json_rules": [],
            "code": [],
            "clarification": None,
            "matched_fields": {}
        }

        for tool_call in tool_calls:
            func = tool_call.get("function", {})
            tool_name = func.get("name", "")
            arguments = func.get("arguments", {})

            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {}

            logger.debug(f"Executing tool: {tool_name}")

            if tool_name == "match_fields":
                matches = self._execute_match_fields(arguments)
                result["matched_fields"] = matches
                if matches and not content:
                    result["message"] = self._format_field_matches(matches)

            elif tool_name == "generate_rule":
                rule_result = self._execute_generate_rule(arguments)
                if rule_result.get("type") == "json":
                    result["json_rules"].append(rule_result["rule"])
                else:
                    result["rules"].append(rule_result["rule"])
                if not content:
                    result["message"] = f"Created rule: {rule_result['rule'].get('rule_name', 'New Rule')}"

            elif tool_name == "generate_custom_function":
                code = self._execute_generate_code(arguments)
                result["code"].append(code)
                if not content:
                    result["message"] = "Generated custom validation function."

            elif tool_name == "ask_clarification":
                result["clarification"] = arguments
                self.state.awaiting_clarification = True
                self.state.clarification_context = arguments
                if not content:
                    question = arguments.get("question", "Could you provide more details?")
                    options = arguments.get("options", [])
                    if options:
                        result["message"] = f"{question}\n\nOptions:\n" + "\n".join(f"- {o}" for o in options)
                    else:
                        result["message"] = question

        if not tool_calls and not content:
            result["message"] = "I'm ready to help create data quality rules. Please describe what validation you need, including which field to validate."

        return result

    def _execute_match_fields(self, args: Dict) -> Dict[str, List]:
        """Execute field matching."""
        field_names = args.get("field_names", [])
        matches = self.field_matcher.match_multiple(field_names)

        for query, results in matches.items():
            if results:
                self.state.matched_fields[query] = results[0][0]

        return matches

    def _execute_generate_rule(self, args: Dict) -> Dict:
        """Execute rule generation for CSV or JSON rules."""
        rule_type = args.get("rule_type", "REGEX")
        field_name = args.get("field_name", "")
        validation_value = args.get("validation_value", "")
        error_message = args.get("error_message", "Validation failed")
        severity = args.get("severity", "ERROR")

        # Resolve field name
        if field_name and '.' not in field_name:
            matches = self.field_matcher.match(field_name, limit=1)
            if matches:
                field_name = matches[0][0]

        # Check for JSON rule types
        if rule_type.startswith("JSON_") or rule_type.lower() in ["composite", "conditional", "cross_field"]:
            return self._create_json_rule(args)

        # Create CSV rule based on type
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
        elif rule_type == "LENGTH":
            try:
                parts = validation_value.split(",")
                min_len = int(parts[0]) if parts else 0
                max_len = int(parts[1]) if len(parts) > 1 else 255
            except (ValueError, IndexError):
                min_len, max_len = 0, 255

            rule = self.rule_generator.create_length_rule(
                field_name=field_name,
                min_length=min_len,
                max_length=max_len,
                error_message=error_message,
                severity=severity
            )
        elif rule_type == "RANGE":
            try:
                parts = validation_value.split(",")
                min_val = float(parts[0]) if parts else 0
                max_val = float(parts[1]) if len(parts) > 1 else 100
            except (ValueError, IndexError):
                min_val, max_val = 0, 100

            rule = self.rule_generator.create_range_rule(
                field_name=field_name,
                min_value=min_val,
                max_value=max_val,
                error_message=error_message,
                severity=severity
            )
        elif rule_type == "IN_LIST":
            allowed = [v.strip() for v in validation_value.split(",")] if validation_value else []
            rule = self.rule_generator.create_in_list_rule(
                field_name=field_name,
                allowed_values=allowed,
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
                pattern=validation_value or ".*",
                error_message=error_message,
                severity=severity
            )

        self.state.generated_rules.append(rule)
        return {"type": "csv", "rule": self.rule_generator.to_json(rule)}

    def _create_json_rule(self, args: Dict) -> Dict:
        """Create a JSON format rule for complex validations."""
        rule_type = args.get("rule_type", "composite").lower()
        error_message = args.get("error_message", "Validation failed")
        severity = args.get("severity", "ERROR")

        if "composite" in rule_type:
            json_rule = {
                "type": "composite",
                "operator": args.get("operator", "AND"),
                "rules": args.get("rules", []),
                "error_message": error_message,
                "severity": severity
            }
        elif "conditional" in rule_type:
            json_rule = {
                "type": "conditional",
                "if": {
                    "field": args.get("condition_field", ""),
                    "equals": args.get("condition_value", "")
                },
                "then": {
                    "field": args.get("then_field", ""),
                    "validation": args.get("then_validation", "NOT_NULL")
                },
                "error_message": error_message,
                "severity": severity
            }
        elif "cross_field" in rule_type:
            json_rule = {
                "type": "cross_field",
                "validation": args.get("comparison", "LESS_THAN"),
                "field1": args.get("field1", ""),
                "field2": args.get("field2", ""),
                "error_message": error_message,
                "severity": severity
            }
        else:
            json_rule = args

        self.state.generated_json_rules.append(json_rule)
        return {"type": "json", "rule": json_rule}

    def _execute_generate_code(self, args: Dict) -> str:
        """Execute code generation."""
        func_name = args.get("function_name", "custom_validation")
        description = args.get("description", "Custom validation")
        params = args.get("parameters", [])

        desc_lower = description.lower()

        if any(word in desc_lower for word in ["exclamation", "special char", "character", "contains"]):
            chars = "!" if "exclamation" in desc_lower else "!"
            func = self.code_generator.generate_no_chars_function(
                name=func_name,
                disallowed_chars=chars,
                error_message=f"Value contains disallowed character(s): {chars}"
            )
        elif "url" in desc_lower:
            func = self.code_generator.generate_url_validation_function(
                name=func_name,
                require_https="https" in desc_lower
            )
        elif "date" in desc_lower and ("before" in desc_lower or "after" in desc_lower):
            comparison = "before" if "before" in desc_lower else "after"
            func = self.code_generator.generate_date_comparison_function(
                name=func_name,
                comparison=comparison
            )
        else:
            func = self.code_generator.generate_function(
                name=func_name,
                description=description,
                validation_logic="""# Implement validation logic
# Return (True, None) if valid
# Return (False, "error message") if invalid
return True, None""",
                parameters=params
            )

        code = self.code_generator.get_full_code(func)
        self.state.generated_code.append(code)
        return code

    def _format_field_matches(self, matches: Dict[str, List]) -> str:
        """Format field match results."""
        lines = ["Found matching fields:"]
        for query, results in matches.items():
            if results:
                top = results[0]
                lines.append(f"- '{query}' -> `{top[0]}` ({top[1]}% match)")
            else:
                lines.append(f"- '{query}' -> No match found")
        return "\n".join(lines)

    def _fallback_processing(self, user_input: str, error: str) -> Dict[str, Any]:
        """Fallback when LLM fails."""
        logger.warning(f"Using fallback processing: {error}")

        result = {
            "message": "",
            "rules": [],
            "json_rules": [],
            "code": [],
            "clarification": None,
            "matched_fields": {}
        }

        input_lower = user_input.lower()

        # Extract fields
        potential_fields = re.findall(r'\b(\w+(?:_\w+)+)\b', user_input)
        for pf in potential_fields:
            matches = self.field_matcher.match(pf, limit=3)
            if matches:
                result["matched_fields"][pf] = matches

        # Detect and generate rule
        rule_type = self.validator.detect_rule_type(user_input)

        if rule_type == RuleType.REGEX and potential_fields:
            field = potential_fields[0]
            matched = self.field_matcher.match(field, limit=1)
            if matched:
                field = matched[0][0]

            pattern = "^[^!]*$" if "exclamation" in input_lower else ".*"
            msg = "contains exclamation point" if "exclamation" in input_lower else "failed validation"

            rule = self.rule_generator.create_regex_rule(
                field_name=field,
                pattern=pattern,
                error_message=f"Value {msg}",
                severity="ERROR"
            )
            self.state.generated_rules.append(rule)
            result["rules"].append(self.rule_generator.to_json(rule))
            result["message"] = f"Created REGEX rule for '{field}'"

        elif rule_type == RuleType.NOT_NULL and potential_fields:
            field = potential_fields[0]
            matched = self.field_matcher.match(field, limit=1)
            if matched:
                field = matched[0][0]

            rule = self.rule_generator.create_not_null_rule(
                field_name=field,
                error_message=f"{field} is required"
            )
            self.state.generated_rules.append(rule)
            result["rules"].append(self.rule_generator.to_json(rule))
            result["message"] = f"Created NOT_NULL rule for '{field}'"

        elif not potential_fields:
            result["message"] = "I need to know which field to validate. Which field should this rule apply to?"
            result["clarification"] = {"question": "Which field should this rule apply to?"}

        else:
            result["message"] = f"What type of validation do you need for '{potential_fields[0]}'?"

        return result

    def get_rules_csv(self) -> str:
        """Get all generated CSV rules."""
        if not self.state.generated_rules:
            return ""

        filepath = self.rule_generator.save_rules(self.state.generated_rules)
        with open(filepath, 'r') as f:
            return f.read()

    def get_json_rules(self) -> str:
        """Get all generated JSON rules."""
        if not self.state.generated_json_rules:
            return ""
        return json.dumps(self.state.generated_json_rules, indent=2)

    def reset(self):
        """Reset the conversation state."""
        self.state = ConversationState()
        self.state.messages.append(Message(
            role="system",
            content=self.SYSTEM_PROMPT
        ))
        logger.info("Conversation reset")
