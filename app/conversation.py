"""
Conversation management for multi-turn dialogue with the LLM.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import json
import re

from .llm_client import OllamaClient, Message, TOOLS
from .field_matcher import FieldMatcher
from .rule_generator import RuleGenerator
from .code_generator import CodeGenerator


@dataclass
class ConversationState:
    """Tracks the state of the conversation."""
    messages: List[Message] = field(default_factory=list)
    pending_fields: List[str] = field(default_factory=list)
    matched_fields: Dict[str, str] = field(default_factory=dict)
    generated_rules: List[Any] = field(default_factory=list)
    generated_code: List[str] = field(default_factory=list)
    awaiting_clarification: bool = False
    clarification_context: Optional[Dict] = None


class ConversationManager:
    """Manages multi-turn conversations for rule creation."""

    SYSTEM_PROMPT = """You are a data quality rules assistant. Help users create validation rules for the ORBIS data dictionary.

Your capabilities:
1. Match field names using fuzzy matching (call match_fields tool)
2. Generate validation rules in CSV format (call generate_rule tool)
3. Create custom Python validation functions (call generate_custom_function tool)
4. Ask clarifying questions when needed (call ask_clarification tool)

Available rule types:
- REGEX: Pattern matching validation (use for format checks, character restrictions)
- NOT_NULL: Required field check
- LENGTH: String length validation (min,max)
- RANGE: Numeric range validation (min,max)
- IN_LIST: Allowed values list
- CUSTOM_FUNCTION: Complex validation requiring Python code

When the user describes a rule:
1. Identify the field(s) they're referring to - use match_fields if uncertain
2. Determine the appropriate rule type
3. If information is missing (field name, severity, exact requirement), use ask_clarification
4. Generate the rule using generate_rule tool
5. If the validation is complex, also generate Python code with generate_custom_function

For checking if a field contains specific characters (like exclamation points):
- Use REGEX with a pattern like ^[^!]*$ (matches strings without the character)
- Or generate a custom function for complex character checks

Always be helpful even with typos or informal language. Infer what the user means."""

    def __init__(self):
        """Initialize the conversation manager."""
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
        """
        Process a user message and return the response.

        Args:
            user_input: The user's message

        Returns:
            Dictionary with 'message', 'rules', 'code', 'clarification', 'matched_fields'
        """
        # Add user message to history
        self.state.messages.append(Message(role="user", content=user_input))

        try:
            # Try to call LLM with tools
            response = self.llm.chat(
                messages=self.state.messages,
                tools=TOOLS
            )

            # Process the response
            result = self._process_response(response)

            # Add assistant response to history
            if result.get("message"):
                self.state.messages.append(Message(
                    role="assistant",
                    content=result["message"]
                ))

            return result

        except Exception as e:
            # Fallback: Try simple parsing without tool calling
            return self._fallback_processing(user_input, str(e))

    def _process_response(self, response: Dict) -> Dict[str, Any]:
        """Process the LLM response and execute tool calls."""
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

        # Execute tool calls
        for tool_call in tool_calls:
            func = tool_call.get("function", {})
            tool_name = func.get("name", "")
            arguments = func.get("arguments", {})

            # Parse arguments if they're a string
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {}

            if tool_name == "match_fields":
                matches = self._execute_match_fields(arguments)
                result["matched_fields"] = matches
                # Add context to message
                if matches and not content:
                    result["message"] = self._format_field_matches(matches)

            elif tool_name == "generate_rule":
                rule = self._execute_generate_rule(arguments)
                result["rules"].append(rule)
                if not content:
                    result["message"] = f"Generated rule: {rule.get('rule_name', 'Unknown')}"

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
                    result["message"] = arguments.get("question", "Could you please provide more details?")

        # If no tool calls and no content, provide a default response
        if not tool_calls and not content:
            result["message"] = "I understand you want to create a rule. Could you describe what validation you need?"

        return result

    def _execute_match_fields(self, args: Dict) -> Dict[str, List]:
        """Execute field matching."""
        field_names = args.get("field_names", [])
        matches = self.field_matcher.match_multiple(field_names)

        # Store best matches
        for query, results in matches.items():
            if results:
                self.state.matched_fields[query] = results[0][0]

        return matches

    def _execute_generate_rule(self, args: Dict) -> Dict:
        """Execute rule generation."""
        rule_type = args.get("rule_type", "REGEX")
        field_name = args.get("field_name", "")
        validation_value = args.get("validation_value", "")
        error_message = args.get("error_message", "Validation failed")
        severity = args.get("severity", "ERROR")

        # Try to resolve field name if not fully qualified
        if field_name and '.' not in field_name:
            matches = self.field_matcher.match(field_name, limit=1)
            if matches:
                field_name = matches[0][0]

        # Create appropriate rule type
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
        elif rule_type == "RANGE":
            try:
                parts = validation_value.split(",")
                min_val = float(parts[0]) if len(parts) > 0 else 0
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
        elif rule_type == "CUSTOM_FUNCTION":
            rule = self.rule_generator.create_custom_function_rule(
                field_name=field_name,
                function_name=validation_value,
                error_message=error_message,
                severity=severity
            )
        else:
            # Default to regex
            rule = self.rule_generator.create_regex_rule(
                field_name=field_name,
                pattern=validation_value or ".*",
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

        # Determine what kind of function to generate based on description
        desc_lower = description.lower()

        if any(word in desc_lower for word in ["exclamation", "special char", "character", "contains"]):
            # Character check function
            # Extract the character(s) to check
            if "exclamation" in desc_lower:
                chars = "!"
            else:
                # Try to find quoted characters
                match = re.search(r"['\"](.+?)['\"]", description)
                chars = match.group(1) if match else "!"

            func = self.code_generator.generate_no_chars_function(
                name=func_name,
                disallowed_chars=chars,
                error_message=f"Value contains disallowed character(s): {chars}"
            )
        elif "url" in desc_lower:
            func = self.code_generator.generate_url_validation_function(
                name=func_name,
                require_https="https" in desc_lower and "http" not in desc_lower.replace("https", "")
            )
        elif "date" in desc_lower and ("before" in desc_lower or "after" in desc_lower):
            comparison = "before" if "before" in desc_lower else "after"
            func = self.code_generator.generate_date_comparison_function(
                name=func_name,
                comparison=comparison
            )
        elif "regex" in desc_lower or "pattern" in desc_lower:
            # Try to extract pattern
            match = re.search(r"['\"](.+?)['\"]", description)
            pattern = match.group(1) if match else ".*"
            func = self.code_generator.generate_regex_function(
                name=func_name,
                pattern=pattern,
                error_message="Value does not match required pattern"
            )
        else:
            # Generic function template
            func = self.code_generator.generate_function(
                name=func_name,
                description=description,
                validation_logic="""# TODO: Implement validation logic
# Return (True, None) if valid
# Return (False, "error message") if invalid
return True, None""",
                parameters=params
            )

        code = self.code_generator.get_full_code(func)
        self.state.generated_code.append(code)
        return code

    def _format_field_matches(self, matches: Dict[str, List]) -> str:
        """Format field match results as a message."""
        lines = ["I found these matching fields:"]
        for query, results in matches.items():
            if results:
                top_match = results[0]
                lines.append(f"- '{query}' -> `{top_match[0]}` (confidence: {top_match[1]}%)")
            else:
                lines.append(f"- '{query}' -> No match found")
        return "\n".join(lines)

    def _fallback_processing(self, user_input: str, error: str) -> Dict[str, Any]:
        """Fallback processing when LLM tool calling fails."""
        result = {
            "message": "",
            "rules": [],
            "code": [],
            "clarification": None,
            "matched_fields": {}
        }

        input_lower = user_input.lower()

        # Try to extract field names
        potential_fields = re.findall(r'\b(\w+(?:_\w+)+)\b', user_input)
        if potential_fields:
            for pf in potential_fields:
                matches = self.field_matcher.match(pf, limit=3)
                if matches:
                    result["matched_fields"][pf] = matches

        # Detect rule intent
        if any(word in input_lower for word in ["rule", "check", "validate", "ensure"]):
            # Try to create a rule
            if "exclamation" in input_lower or "!" in user_input:
                # Character check rule
                field = potential_fields[0] if potential_fields else "unknown_field"
                matched = self.field_matcher.match(field, limit=1)
                if matched:
                    field = matched[0][0]

                rule = self.rule_generator.create_regex_rule(
                    field_name=field,
                    pattern="^[^!]*$",
                    error_message="Value contains exclamation point",
                    severity="ERROR"
                )
                self.state.generated_rules.append(rule)
                result["rules"].append(self.rule_generator.to_json(rule))
                result["message"] = f"Created a rule to check '{field}' for exclamation points."

            elif any(word in input_lower for word in ["not empty", "not blank", "required", "not null"]):
                field = potential_fields[0] if potential_fields else "unknown_field"
                matched = self.field_matcher.match(field, limit=1)
                if matched:
                    field = matched[0][0]

                rule = self.rule_generator.create_not_null_rule(
                    field_name=field,
                    error_message=f"{field} is required"
                )
                self.state.generated_rules.append(rule)
                result["rules"].append(self.rule_generator.to_json(rule))
                result["message"] = f"Created a NOT_NULL rule for '{field}'."

            else:
                result["message"] = "I understand you want to create a rule. Could you specify:\n1. Which field should be validated?\n2. What validation should be applied?\n3. What should happen if it fails (ERROR/WARNING)?"

        else:
            result["message"] = f"I'm ready to help create data quality rules. Describe what you'd like to validate.\n\n(Note: Using fallback mode due to: {error})"

        return result

    def get_rules_csv(self) -> str:
        """Get all generated rules as CSV content."""
        if not self.state.generated_rules:
            return ""

        filepath = self.rule_generator.save_rules(self.state.generated_rules)
        with open(filepath, 'r') as f:
            return f.read()

    def reset(self):
        """Reset the conversation state."""
        self.state = ConversationState()
        self.state.messages.append(Message(
            role="system",
            content=self.SYSTEM_PROMPT
        ))
