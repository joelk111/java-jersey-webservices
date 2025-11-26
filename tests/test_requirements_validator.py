"""
Unit tests for the RequirementsValidator module.
Tests detection of missing required fields for rule creation.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.requirements_validator import RequirementsValidator, RuleType


@pytest.fixture
def validator():
    """Create a RequirementsValidator instance."""
    return RequirementsValidator()


class TestRuleTypeDetection:
    """Test automatic rule type detection from user input."""

    def test_detect_regex_pattern(self, validator):
        """Test detecting REGEX rule from pattern keywords."""
        inputs = [
            "check if field contains exclamation",
            "validate the pattern matches",
            "ensure format is correct",
            "check for special characters",
        ]
        for text in inputs:
            rule_type = validator.detect_rule_type(text)
            assert rule_type == RuleType.REGEX, f"Failed for: {text}"

    def test_detect_not_null(self, validator):
        """Test detecting NOT_NULL rule."""
        inputs = [
            "field should not be null",
            "field is required",
            "ensure field is not empty",
            "field must have a value",
            "field cannot be blank",
        ]
        for text in inputs:
            rule_type = validator.detect_rule_type(text)
            assert rule_type == RuleType.NOT_NULL, f"Failed for: {text}"

    def test_detect_length(self, validator):
        """Test detecting LENGTH rule."""
        inputs = [
            "field must be 2 characters",
            "ensure length is correct",
            "must be exactly 5 chars long",
        ]
        for text in inputs:
            rule_type = validator.detect_rule_type(text)
            assert rule_type == RuleType.LENGTH, f"Failed for: {text}"

    def test_detect_range(self, validator):
        """Test detecting RANGE rule."""
        inputs = [
            "value must be between 0 and 100",
            "number should be at least 10",
            "percentage range check",
        ]
        for text in inputs:
            rule_type = validator.detect_rule_type(text)
            assert rule_type == RuleType.RANGE, f"Failed for: {text}"

    def test_detect_in_list(self, validator):
        """Test detecting IN_LIST rule."""
        inputs = [
            "value must be one of Active, Inactive",
            "check allowed values",
            "field options are A, B, C",
        ]
        for text in inputs:
            rule_type = validator.detect_rule_type(text)
            assert rule_type == RuleType.IN_LIST, f"Failed for: {text}"

    def test_detect_custom_function(self, validator):
        """Test detecting CUSTOM_FUNCTION rule."""
        inputs = [
            "need a custom validation",
            "use special algorithm to check",
            "complex validation required",
        ]
        for text in inputs:
            rule_type = validator.detect_rule_type(text)
            assert rule_type == RuleType.CUSTOM_FUNCTION, f"Failed for: {text}"

    def test_detect_unknown(self, validator):
        """Test handling of unrecognizable input."""
        rule_type = validator.detect_rule_type("hello world how are you")
        assert rule_type is None


class TestRequiredFieldValidation:
    """Test validation of required fields."""

    def test_regex_complete(self, validator):
        """Test REGEX validation with all required fields."""
        result = validator.validate_request(
            "check if web_url contains exclamation points",
            {"field_name": "web_url", "pattern": "^[^!]*$"}
        )
        assert result.is_valid
        assert len(result.missing_required) == 0

    def test_regex_missing_field(self, validator):
        """Test REGEX validation missing field_name."""
        result = validator.validate_request(
            "check for exclamation points",
            {"pattern": "^[^!]*$"}
        )
        assert not result.is_valid
        assert "field_name" in result.missing_required

    def test_not_null_complete(self, validator):
        """Test NOT_NULL validation with all required fields."""
        result = validator.validate_request(
            "make sure email_address is not empty",
            {"field_name": "email_address"}
        )
        assert result.is_valid

    def test_not_null_missing_field(self, validator):
        """Test NOT_NULL validation missing field_name."""
        result = validator.validate_request(
            "make sure field is not empty",
            {}
        )
        assert not result.is_valid
        assert "field_name" in result.missing_required

    def test_length_complete(self, validator):
        """Test LENGTH validation with all required fields."""
        result = validator.validate_request(
            "country code must be exactly 2 characters",
            {"field_name": "country_code", "min_length": "2", "max_length": "2"}
        )
        assert result.is_valid

    def test_range_missing_values(self, validator):
        """Test RANGE validation missing min/max."""
        result = validator.validate_request(
            "percentage must be in range for percentage_field",
            {"field_name": "percentage_field"}
        )
        assert not result.is_valid
        assert "min_value" in result.missing_required or "max_value" in result.missing_required


class TestClarificationMessages:
    """Test clarification message generation."""

    def test_missing_field_message(self, validator):
        """Test clarification message for missing field."""
        result = validator.validate_request(
            "check if email is null",
            {}
        )
        message = validator.format_missing_fields_message(result)
        # Should ask about field since NOT_NULL was detected
        assert "field" in message.lower() or "which" in message.lower()

    def test_no_rule_type_message(self, validator):
        """Test message when rule type is unclear."""
        result = validator.validate_request(
            "do something with data",
            {}
        )
        message = validator.format_missing_fields_message(result)
        assert len(message) > 0

    def test_valid_request_no_message(self, validator):
        """Test no message for valid request."""
        result = validator.validate_request(
            "check if email_address is not empty",
            {"field_name": "email_address"}
        )
        message = validator.format_missing_fields_message(result)
        assert message == ""


class TestRequirementsSummary:
    """Test requirements summary functionality."""

    def test_get_summary(self, validator):
        """Test getting requirements summary for a rule type."""
        summary = validator.get_requirements_summary(RuleType.REGEX)
        assert "REGEX" in summary
        assert "field_name" in summary
        assert "pattern" in summary

    def test_all_rule_types_have_summary(self, validator):
        """Test all rule types have valid summaries."""
        for rule_type in RuleType:
            summary = validator.get_requirements_summary(rule_type)
            assert len(summary) > 0
            assert rule_type.value in summary


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_empty_input(self, validator):
        """Test handling of empty input."""
        result = validator.validate_request("", {})
        # Should not crash
        assert result is not None

    def test_whitespace_only(self, validator):
        """Test handling of whitespace-only input."""
        result = validator.validate_request("   ", {})
        assert result is not None

    def test_special_characters_in_input(self, validator):
        """Test handling of special characters."""
        result = validator.validate_request(
            "check if field has !@#$%^&*()",
            {"field_name": "test_field"}
        )
        assert result.detected_rule_type == RuleType.REGEX
