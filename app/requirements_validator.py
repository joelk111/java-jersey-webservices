"""
Requirements validator for rule creation.

This module defines what information is REQUIRED for each rule type
and validates that all required fields are present before generating rules.
"""

from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum


class RuleType(Enum):
    """Supported rule types."""
    REGEX = "REGEX"
    NOT_NULL = "NOT_NULL"
    LENGTH = "LENGTH"
    RANGE = "RANGE"
    IN_LIST = "IN_LIST"
    CUSTOM_FUNCTION = "CUSTOM_FUNCTION"
    JSON_COMPOSITE = "JSON_COMPOSITE"
    JSON_CONDITIONAL = "JSON_CONDITIONAL"
    JSON_CROSS_FIELD = "JSON_CROSS_FIELD"


@dataclass
class RuleRequirements:
    """Defines required and optional fields for a rule type."""
    rule_type: RuleType
    required_fields: Set[str]
    optional_fields: Set[str]
    description: str
    example_prompt: str


# Define requirements for each rule type
RULE_REQUIREMENTS: Dict[RuleType, RuleRequirements] = {
    RuleType.REGEX: RuleRequirements(
        rule_type=RuleType.REGEX,
        required_fields={"field_name", "pattern"},
        optional_fields={"error_message", "severity", "rule_name"},
        description="Validates field value against a regular expression pattern",
        example_prompt="Check if web_url contains exclamation points"
    ),
    RuleType.NOT_NULL: RuleRequirements(
        rule_type=RuleType.NOT_NULL,
        required_fields={"field_name"},
        optional_fields={"error_message", "severity", "rule_name"},
        description="Ensures a field has a non-null/non-empty value",
        example_prompt="Make sure email_address is not empty"
    ),
    RuleType.LENGTH: RuleRequirements(
        rule_type=RuleType.LENGTH,
        required_fields={"field_name", "min_length", "max_length"},
        optional_fields={"error_message", "severity", "rule_name"},
        description="Validates string length is within specified bounds",
        example_prompt="Country code must be exactly 2 characters"
    ),
    RuleType.RANGE: RuleRequirements(
        rule_type=RuleType.RANGE,
        required_fields={"field_name", "min_value", "max_value"},
        optional_fields={"error_message", "severity", "rule_name"},
        description="Validates numeric value is within specified range",
        example_prompt="Percentage must be between 0 and 100"
    ),
    RuleType.IN_LIST: RuleRequirements(
        rule_type=RuleType.IN_LIST,
        required_fields={"field_name", "allowed_values"},
        optional_fields={"error_message", "severity", "rule_name"},
        description="Validates value is in a list of allowed values",
        example_prompt="Status must be Active, Inactive, or Pending"
    ),
    RuleType.CUSTOM_FUNCTION: RuleRequirements(
        rule_type=RuleType.CUSTOM_FUNCTION,
        required_fields={"field_name", "function_name", "validation_description"},
        optional_fields={"error_message", "severity", "rule_name", "parameters"},
        description="Uses a custom Python function for complex validation",
        example_prompt="Validate BVD ID checksum using custom algorithm"
    ),
    RuleType.JSON_COMPOSITE: RuleRequirements(
        rule_type=RuleType.JSON_COMPOSITE,
        required_fields={"fields", "operator"},  # AND/OR
        optional_fields={"error_message", "severity", "rule_name"},
        description="Combines multiple validations with AND/OR logic",
        example_prompt="Address must have city AND country AND postcode"
    ),
    RuleType.JSON_CONDITIONAL: RuleRequirements(
        rule_type=RuleType.JSON_CONDITIONAL,
        required_fields={"condition_field", "condition_value", "then_field", "then_validation"},
        optional_fields={"error_message", "severity", "rule_name"},
        description="Conditional validation: if X then validate Y",
        example_prompt="If has_email is Y, then email must be valid"
    ),
    RuleType.JSON_CROSS_FIELD: RuleRequirements(
        rule_type=RuleType.JSON_CROSS_FIELD,
        required_fields={"field1", "field2", "comparison"},  # <, >, =, !=
        optional_fields={"error_message", "severity", "rule_name"},
        description="Compares two fields against each other",
        example_prompt="Start date must be before end date"
    ),
}

# Fields that are NEVER required (auto-generated)
AUTO_GENERATED_FIELDS = {"rule_id"}

# Default values for optional fields
DEFAULT_VALUES = {
    "severity": "ERROR",
    "error_message": "Validation failed",
}


@dataclass
class ValidationResult:
    """Result of validating rule requirements."""
    is_valid: bool
    missing_required: List[str]
    detected_rule_type: Optional[RuleType]
    extracted_values: Dict[str, str]
    clarification_needed: List[str]


class RequirementsValidator:
    """Validates that all required information is present for rule creation."""

    def __init__(self):
        self.requirements = RULE_REQUIREMENTS

    def detect_rule_type(self, user_input: str) -> Optional[RuleType]:
        """
        Detect the likely rule type from user input.

        Args:
            user_input: The user's natural language request

        Returns:
            The detected RuleType or None if unclear
        """
        # Normalize whitespace and convert to lowercase
        input_lower = ' '.join(user_input.lower().split())

        # Check for standard rule types FIRST (most common cases)

        # REGEX patterns - check before other rules
        if any(word in input_lower for word in ["pattern", "regex", "match", "format", "contains", "exclamation", "special char", "!@#"]):
            return RuleType.REGEX
        # Also check for "has" followed by special character symbols
        if " has " in input_lower and any(c in input_lower for c in "!@#$%^&*"):
            return RuleType.REGEX

        # NOT_NULL - check for null/empty/required before conditional
        if any(phrase in input_lower for phrase in [
            "not null", "not empty", "is required", "must have", "not blank",
            "mandatory", "is not empty", "is not null", "cannot be empty",
            "cannot be null", "can't be empty", "can't be null", "gotta have",
            "aint empty", "ain't empty", "needs to have", "should not be null",
            "should not be empty", "cannot be blank", "must not be blank",
            "must not be null", "must not be empty", "if null", "is null",
            "checks if null", "check null", "null check"
        ]):
            return RuleType.NOT_NULL

        # LENGTH - check for length-related phrases
        if any(word in input_lower for word in ["length", "characters", "chars long", "exactly"]):
            return RuleType.LENGTH

        # RANGE - numeric ranges
        if any(word in input_lower for word in ["between", "range", "minimum", "maximum", "at least", "at most"]):
            if any(word in input_lower for word in ["number", "value", "percent", "amount", "0", "1"]):
                return RuleType.RANGE

        # IN_LIST - allowed values
        if any(phrase in input_lower for phrase in ["must be one of", "allowed values", "in list", "options are"]):
            return RuleType.IN_LIST

        # CUSTOM_FUNCTION - complex validations
        if any(word in input_lower for word in ["custom", "algorithm", "complex", "special validation"]):
            return RuleType.CUSTOM_FUNCTION

        # JSON rules - check last, with more specific patterns
        # JSON_COMPOSITE - requires multiple AND conditions
        if " and " in input_lower and input_lower.count(" and ") >= 2:
            return RuleType.JSON_COMPOSITE

        # JSON_CONDITIONAL - requires "if...then" pattern, not just "check if"
        if ("if " in input_lower and " then " in input_lower) or "conditional" in input_lower:
            return RuleType.JSON_CONDITIONAL

        # JSON_CROSS_FIELD - comparing two fields
        if any(phrase in input_lower for phrase in ["before", "after", "greater than", "less than", "compare"]):
            if any(word in input_lower for word in ["date", "field1", "field2", "compare field"]):
                return RuleType.JSON_CROSS_FIELD

        return None

    def validate_request(
        self,
        user_input: str,
        extracted_info: Dict[str, str]
    ) -> ValidationResult:
        """
        Validate that a rule creation request has all required information.

        Args:
            user_input: The user's natural language request
            extracted_info: Information already extracted from the request

        Returns:
            ValidationResult with validation status and missing fields
        """
        # Detect rule type
        rule_type = self.detect_rule_type(user_input)

        if rule_type is None:
            return ValidationResult(
                is_valid=False,
                missing_required=["rule_type"],
                detected_rule_type=None,
                extracted_values=extracted_info,
                clarification_needed=[
                    "What type of validation do you need? (format check, required field, range check, etc.)"
                ]
            )

        # Get requirements for this rule type
        requirements = self.requirements[rule_type]

        # Check which required fields are missing
        missing = []
        clarifications = []

        for field in requirements.required_fields:
            if field not in extracted_info or not extracted_info[field]:
                missing.append(field)

                # Generate specific clarification question
                clarification = self._get_clarification_question(field, rule_type)
                if clarification:
                    clarifications.append(clarification)

        return ValidationResult(
            is_valid=len(missing) == 0,
            missing_required=missing,
            detected_rule_type=rule_type,
            extracted_values=extracted_info,
            clarification_needed=clarifications
        )

    def _get_clarification_question(self, field: str, rule_type: RuleType) -> str:
        """Generate a user-friendly clarification question for a missing field."""
        questions = {
            "field_name": "Which field should this rule apply to?",
            "pattern": "What pattern should the value match? (e.g., no exclamation points, starts with http)",
            "min_length": "What is the minimum length allowed?",
            "max_length": "What is the maximum length allowed?",
            "min_value": "What is the minimum value allowed?",
            "max_value": "What is the maximum value allowed?",
            "allowed_values": "What values are allowed? (comma-separated list)",
            "function_name": "What should I name the custom validation function?",
            "validation_description": "Describe what the custom validation should check",
            "fields": "Which fields should be validated together?",
            "operator": "Should all conditions pass (AND) or any condition (OR)?",
            "condition_field": "Which field triggers the conditional check?",
            "condition_value": "What value should trigger the validation?",
            "then_field": "Which field should be validated when the condition is met?",
            "then_validation": "What validation should be applied?",
            "field1": "What is the first field to compare?",
            "field2": "What is the second field to compare?",
            "comparison": "How should they be compared? (before, after, equal, not equal)",
        }
        return questions.get(field, f"Please provide the {field}")

    def get_requirements_summary(self, rule_type: RuleType) -> str:
        """Get a human-readable summary of requirements for a rule type."""
        req = self.requirements[rule_type]
        return f"""
Rule Type: {rule_type.value}
Description: {req.description}
Required: {', '.join(req.required_fields)}
Optional: {', '.join(req.optional_fields)}
Example: "{req.example_prompt}"
"""

    def format_missing_fields_message(self, result: ValidationResult) -> str:
        """Format a user-friendly message about missing required fields."""
        if result.is_valid:
            return ""

        if result.detected_rule_type is None:
            return "I'm not sure what type of rule you want to create. Could you describe the validation you need?"

        lines = [
            f"I understand you want to create a {result.detected_rule_type.value} rule.",
            "However, I need a bit more information:",
            ""
        ]

        for i, clarification in enumerate(result.clarification_needed, 1):
            lines.append(f"{i}. {clarification}")

        return "\n".join(lines)
