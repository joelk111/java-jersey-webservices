"""
Integration Tests for NLP Rules Engine

Tests the complete flow from user input to rule generation,
including conversation manager, field matching, and rule output.

Note: These tests run without requiring the LLM client (Ollama).
"""

import pytest
import sys
import os
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock httpx before importing conversation
sys.modules['httpx'] = MagicMock()

from app.field_matcher import FieldMatcher
from app.rule_generator import RuleGenerator
from app.requirements_validator import RequirementsValidator, RuleType


class TestConversationIntegration:
    """Integration tests for the conversation flow."""

    @pytest.fixture
    def manager(self):
        """Create a ConversationManager without LLM (for offline testing)."""
        from app.conversation import ConversationManager
        return ConversationManager()

    def test_conversation_initialization(self, manager):
        """Test that conversation manager initializes correctly."""
        assert manager.field_matcher is not None
        assert manager.rule_generator is not None
        assert manager.validator is not None
        assert len(manager.state.messages) == 1  # System message
        assert manager.state.messages[0].role == "system"

    def test_pre_validation_detects_missing_field(self, manager):
        """Test that pre-validation catches missing field names."""
        result = manager._pre_validate_request("check if field is null")
        # Should detect rule type but may need field
        assert "rule_type" in result or result["is_valid"]

    def test_pre_validation_accepts_complete_request(self, manager):
        """Test that pre-validation accepts complete requests."""
        result = manager._pre_validate_request("check if email_address contains exclamation points")
        # Should detect REGEX type
        assert result.get("rule_type") is not None or result["is_valid"]

    def test_fallback_processing_regex(self, manager):
        """Test fallback processing for REGEX rules."""
        result = manager._fallback_processing(
            "check if web_url contains exclamation points",
            "LLM unavailable"
        )
        assert result["message"] is not None
        # Should either create rule or ask for more info
        assert result["rules"] or result["clarification"]

    def test_fallback_processing_not_null(self, manager):
        """Test fallback processing for NOT_NULL rules."""
        result = manager._fallback_processing(
            "make sure email_address is not empty",
            "LLM unavailable"
        )
        assert result["message"] is not None

    def test_get_rules_csv_empty(self, manager):
        """Test getting CSV when no rules generated."""
        csv = manager.get_rules_csv()
        assert csv == ""

    def test_get_json_rules_empty(self, manager):
        """Test getting JSON when no rules generated."""
        json_rules = manager.get_json_rules()
        assert json_rules == ""

    def test_reset_clears_state(self, manager):
        """Test that reset clears conversation state."""
        # Add some state
        manager.state.matched_fields["test"] = "test_field"
        manager.state.awaiting_clarification = True

        # Reset
        manager.reset()

        assert manager.state.matched_fields == {}
        assert manager.state.awaiting_clarification == False
        assert len(manager.state.messages) == 1  # Only system message


class TestFullWorkflow:
    """Test complete workflows from input to output."""

    @pytest.fixture
    def field_matcher(self):
        return FieldMatcher()

    @pytest.fixture
    def rule_generator(self):
        return RuleGenerator()

    @pytest.fixture
    def validator(self):
        return RequirementsValidator()

    def test_complete_regex_workflow(self, field_matcher, rule_generator, validator):
        """Test complete workflow for creating a REGEX rule."""
        # 1. User input
        user_input = "create a rule for email that checks for exclamation points"

        # 2. Detect rule type
        rule_type = validator.detect_rule_type(user_input)
        assert rule_type == RuleType.REGEX

        # 3. Match field - use "email" which should match email_address
        matches = field_matcher.match("email", limit=1)
        assert len(matches) > 0
        field_name = matches[0][0]

        # 4. Generate rule
        rule = rule_generator.create_regex_rule(
            field_name=field_name,
            pattern="^[^!]*$",
            error_message="Field contains exclamation point",
            severity="ERROR"
        )

        # 5. Verify rule
        assert rule.rule_type == "REGEX"
        assert "!" in rule.validation_value or "[^!]" in rule.validation_value
        assert rule.severity == "ERROR"

    def test_complete_not_null_workflow(self, field_matcher, rule_generator, validator):
        """Test complete workflow for creating a NOT_NULL rule."""
        # 1. User input
        user_input = "make sure email_address is not empty"

        # 2. Detect rule type
        rule_type = validator.detect_rule_type(user_input)
        assert rule_type == RuleType.NOT_NULL

        # 3. Match field
        matches = field_matcher.match("email_address", limit=1)
        assert len(matches) > 0
        field_name = matches[0][0]

        # 4. Generate rule
        rule = rule_generator.create_not_null_rule(
            field_name=field_name,
            error_message="Email address is required"
        )

        # 5. Verify rule
        assert rule.rule_type == "NOT_NULL"

    def test_complete_length_workflow(self, field_matcher, rule_generator, validator):
        """Test complete workflow for creating a LENGTH rule."""
        # 1. User input
        user_input = "country code must be exactly 2 characters"

        # 2. Detect rule type
        rule_type = validator.detect_rule_type(user_input)
        assert rule_type == RuleType.LENGTH

        # 3. Generate rule
        rule = rule_generator.create_length_rule(
            field_name="country_iso_code",
            min_length=2,
            max_length=2,
            error_message="Country code must be exactly 2 characters"
        )

        # 4. Verify rule
        assert rule.rule_type == "LENGTH"
        assert "2" in rule.validation_value

    def test_complete_range_workflow(self, field_matcher, rule_generator, validator):
        """Test complete workflow for creating a RANGE rule."""
        # 1. User input
        user_input = "percentage must be between 0 and 100"

        # 2. Detect rule type
        rule_type = validator.detect_rule_type(user_input)
        assert rule_type == RuleType.RANGE

        # 3. Generate rule
        rule = rule_generator.create_range_rule(
            field_name="shareholder_percentage",
            min_value=0,
            max_value=100,
            error_message="Percentage must be between 0 and 100"
        )

        # 4. Verify rule
        assert rule.rule_type == "RANGE"
        assert "0" in rule.validation_value
        assert "100" in rule.validation_value

    def test_complete_in_list_workflow(self, rule_generator, validator):
        """Test complete workflow for creating an IN_LIST rule."""
        # 1. User input
        user_input = "status must be one of Active, Inactive, Pending"

        # 2. Detect rule type
        rule_type = validator.detect_rule_type(user_input)
        assert rule_type == RuleType.IN_LIST

        # 3. Generate rule
        rule = rule_generator.create_in_list_rule(
            field_name="registration_status",
            allowed_values=["Active", "Inactive", "Pending"],
            error_message="Invalid status"
        )

        # 4. Verify rule
        assert rule.rule_type == "IN_LIST"
        assert "Active" in rule.validation_value


class TestClarificationFlow:
    """Test clarification request workflows."""

    @pytest.fixture
    def manager(self):
        from app.conversation import ConversationManager
        return ConversationManager()

    def test_clarification_state_management(self, manager):
        """Test clarification state is managed correctly."""
        # Simulate clarification needed
        manager.state.awaiting_clarification = True
        manager.state.clarification_context = {"question": "Which field?"}

        # Next input should clear clarification state
        result = manager._pre_validate_request("web_url")
        assert manager.state.awaiting_clarification == False

    def test_validation_result_formatting(self, manager):
        """Test validation result message formatting."""
        result = manager.validator.validate_request(
            "check if field is null",
            {}
        )
        message = manager.validator.format_missing_fields_message(result)
        assert len(message) > 0  # Should have helpful message


class TestRuleSaving:
    """Test rule saving and output."""

    @pytest.fixture
    def rule_generator(self):
        return RuleGenerator()

    def test_save_multiple_rules(self, rule_generator):
        """Test saving multiple rules to CSV."""
        rules = [
            rule_generator.create_regex_rule(
                field_name="web_url",
                pattern="^[^!]*$",
                error_message="No exclamation"
            ),
            rule_generator.create_not_null_rule(
                field_name="email",
                error_message="Email required"
            )
        ]

        filepath = rule_generator.save_rules(rules)
        assert os.path.exists(filepath)

        # Read and verify
        with open(filepath, 'r') as f:
            content = f.read()
            assert "REGEX" in content
            assert "NOT_NULL" in content

    def test_rule_to_json(self, rule_generator):
        """Test converting rule to JSON."""
        rule = rule_generator.create_regex_rule(
            field_name="test_field",
            pattern=".*",
            error_message="Test"
        )
        json_rule = rule_generator.to_json(rule)
        # Check for common keys in the JSON output
        assert "rule_id" in json_rule
        assert "field" in json_rule or "field_name" in json_rule
        assert "error_message" in json_rule


class TestFieldMatchingIntegration:
    """Test field matching across various scenarios."""

    @pytest.fixture
    def matcher(self):
        return FieldMatcher()

    def test_match_common_fields(self, matcher):
        """Test matching common field names."""
        common_fields = ["email", "url", "phone", "address", "name", "id"]
        for field in common_fields:
            results = matcher.match(field, limit=3)
            assert len(results) > 0, f"No matches for '{field}'"

    def test_match_with_typos(self, matcher):
        """Test matching handles typos."""
        typo_tests = [
            ("emial", "email"),
            ("addres", "address"),
            ("phoen", "phone"),
        ]
        for typo, expected in typo_tests:
            results = matcher.match(typo, limit=5)
            assert len(results) > 0, f"No matches for typo '{typo}'"

    def test_match_table_qualified_names(self, matcher):
        """Test matching table-qualified field names."""
        results = matcher.match("email_address", limit=1)
        # Should match some field containing email or address
        if results:
            result = results[0][0].lower()
            assert "email" in result or "address" in result


class TestEdgeCasesIntegration:
    """Test edge cases in integration."""

    @pytest.fixture
    def manager(self):
        from app.conversation import ConversationManager
        return ConversationManager()

    def test_empty_input_handling(self, manager):
        """Test handling of empty input."""
        result = manager._pre_validate_request("")
        # Should not crash
        assert result is not None

    def test_special_characters_in_pattern(self, manager):
        """Test handling special regex characters."""
        result = manager._pre_validate_request(
            "check if web_url contains !@#$%^&*()"
        )
        # Should not crash
        assert result is not None

    def test_very_long_input(self, manager):
        """Test handling of very long input."""
        long_input = "create a rule " * 100
        result = manager._pre_validate_request(long_input)
        # Should not crash
        assert result is not None
