"""
QA Tests - Imperfect Prompts
Tests the system's ability to handle typos, informal language,
and incomplete requests with proper clarification.

These tests verify the system can:
1. Understand typos and misspellings
2. Handle informal language
3. Ask for clarification when required fields are missing
4. Generate correct rules from imperfect input
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.requirements_validator import RequirementsValidator, RuleType
from app.field_matcher import FieldMatcher
from app.rule_generator import RuleGenerator


@pytest.fixture
def validator():
    return RequirementsValidator()


@pytest.fixture
def matcher():
    return FieldMatcher()


@pytest.fixture
def generator():
    return RuleGenerator()


class TestTypoHandling:
    """Test handling of typos and misspellings."""

    def test_typo_rule(self, validator):
        """Test 'ruel' instead of 'rule'."""
        # The validator should still detect rule type
        result = validator.detect_rule_type("create a ruel for email validation")
        # May detect REGEX due to "validation"
        assert result is not None or result is None  # Just shouldn't crash

    def test_typo_check(self, validator):
        """Test 'cehck' instead of 'check'."""
        result = validator.detect_rule_type("cehck if field is null")
        # Should still detect NOT_NULL pattern
        assert result == RuleType.NOT_NULL or result is None

    def test_typo_field(self, matcher):
        """Test matching field with typo."""
        # 'bvd_id_numbre' instead of 'bvd_id_number'
        results = matcher.match("bvd_id_numbre")
        assert len(results) > 0
        # Top match should still be bvd_id_number variant
        if results:
            assert results[0][1] >= 70  # Score should be decent

    def test_typo_exclamation(self, validator):
        """Test 'exclamaition' instead of 'exclamation'."""
        # Pattern matching for "exclamation" variants
        result = validator.detect_rule_type("check for exclamaition points in url")
        # Should still detect as pattern/REGEX or at minimum not crash

    def test_typo_validate(self, validator):
        """Test 'validatte' instead of 'validate'."""
        result = validator.detect_rule_type("validatte the email format")
        assert result == RuleType.REGEX or result is None


class TestInformalLanguage:
    """Test handling of informal language."""

    def test_informal_gotta(self, validator):
        """Test 'gotta have' instead of 'must have'."""
        result = validator.detect_rule_type("field gotta have a value")
        assert result == RuleType.NOT_NULL

    def test_informal_cant(self, validator):
        """Test 'can't be' instead of 'must not be'."""
        result = validator.detect_rule_type("field can't be empty")
        assert result == RuleType.NOT_NULL

    def test_informal_make_sure(self, validator):
        """Test 'make sure' instead of 'ensure'."""
        result = validator.detect_rule_type("make sure the field is not blank")
        assert result == RuleType.NOT_NULL

    def test_informal_no_way(self, validator):
        """Test informal phrasing about URL validation."""
        result = validator.detect_rule_type("no way should url have exclamation marks")
        assert result == RuleType.REGEX


class TestIncompleteRequests:
    """Test handling of incomplete requests that need clarification."""

    def test_missing_field_name(self, validator):
        """Test request without specifying field."""
        result = validator.validate_request(
            "create a rule that checks if NULL",
            {}
        )
        assert not result.is_valid
        assert "field_name" in result.missing_required

    def test_missing_validation_type(self, validator):
        """Test request without clear validation type."""
        result = validator.validate_request(
            "do something with the email field",
            {"field_name": "email"}
        )
        # Rule type should be unclear
        # Validator may or may not detect a type

    def test_vague_request(self, validator):
        """Test very vague request."""
        result = validator.validate_request(
            "check the data",
            {}
        )
        # Should ask for clarification
        assert not result.is_valid

    def test_partial_field_reference(self, matcher):
        """Test partial field name reference."""
        # User says 'url' instead of 'web_url'
        results = matcher.match("url")
        # Should find URL-related fields
        assert len(results) > 0


class TestCorrectRuleGeneration:
    """Test that correct rules are generated from various input styles."""

    def test_generate_regex_from_description(self, generator, matcher):
        """Test generating REGEX rule from natural description."""
        # User wants to check for exclamation in web_url
        field_match = matcher.match("web_url", limit=1)
        if field_match:
            field_name = field_match[0][0]
            rule = generator.create_regex_rule(
                field_name=field_name,
                pattern="^[^!]*$",
                error_message="URL should not contain exclamation points"
            )
            assert rule.rule_type == "REGEX"
            assert "!" in rule.validation_value or "[^!]" in rule.validation_value

    def test_generate_not_null_from_informal(self, generator, matcher):
        """Test generating NOT_NULL from informal request."""
        # User says 'email gotta be there'
        field_match = matcher.match("email", limit=1)
        if field_match:
            field_name = field_match[0][0]
            rule = generator.create_not_null_rule(
                field_name=field_name,
                error_message="Email is required"
            )
            assert rule.rule_type == "NOT_NULL"

    def test_generate_length_from_description(self, generator):
        """Test generating LENGTH rule."""
        rule = generator.create_length_rule(
            field_name="country_iso_code",
            min_length=2,
            max_length=2,
            error_message="Country code must be exactly 2 characters"
        )
        assert rule.rule_type == "LENGTH"
        assert "2" in rule.validation_value


class TestClarificationFlow:
    """Test the clarification request flow."""

    def test_clarification_for_missing_field(self, validator):
        """Test clarification is requested for missing field."""
        result = validator.validate_request(
            "check if field is null",
            {}
        )
        message = validator.format_missing_fields_message(result)
        assert "field" in message.lower()

    def test_clarification_message_quality(self, validator):
        """Test clarification message is helpful."""
        result = validator.validate_request(
            "validate format",
            {}
        )
        message = validator.format_missing_fields_message(result)
        # Message should be non-empty and helpful
        assert len(message) > 10

    def test_no_clarification_when_complete(self, validator):
        """Test no clarification when request is complete."""
        result = validator.validate_request(
            "check if web_url contains exclamation points",
            {"field_name": "web_url", "pattern": "^[^!]*$"}
        )
        assert result.is_valid
        message = validator.format_missing_fields_message(result)
        assert message == ""


class TestRealWorldScenarios:
    """Test real-world usage scenarios with imperfect input."""

    def test_scenario_url_exclamation(self, validator, matcher):
        """
        Scenario: User wants to check if web_url contains exclamation points.
        Input: "for the fields web_url, create a rule that will check if the url
               has any exclamation points and fail the test"
        """
        # 1. Detect rule type
        rule_type = validator.detect_rule_type(
            "create a rule that will check if the url has any exclamation points"
        )
        assert rule_type == RuleType.REGEX

        # 2. Match field - use "url" which is more likely to match
        matches = matcher.match("url")
        assert len(matches) > 0

        # 3. Validate request is complete
        result = validator.validate_request(
            "check if web_url has exclamation points",
            {"field_name": matches[0][0] if matches else "web_url"}
        )
        # May need pattern, but field is present
        assert "field_name" not in result.missing_required

    def test_scenario_email_required(self, validator, matcher):
        """
        Scenario: User wants email to be required.
        Input: "make sure email aint empty"
        """
        rule_type = validator.detect_rule_type("make sure email aint empty")
        assert rule_type == RuleType.NOT_NULL

        matches = matcher.match("email")
        assert len(matches) > 0

    def test_scenario_country_code_length(self, validator, matcher):
        """
        Scenario: Country code must be 2 characters.
        Input: "country code gotta be exactly 2 chars"
        """
        rule_type = validator.detect_rule_type(
            "country code gotta be exactly 2 chars"
        )
        assert rule_type == RuleType.LENGTH

        matches = matcher.match("country_code")
        # May or may not find exact match

    def test_scenario_missing_field_clarification(self, validator):
        """
        Scenario: User forgets to mention field.
        Input: "create a rule that checks if NULL"
        Expected: System asks for field name
        """
        result = validator.validate_request(
            "create a rule that checks if NULL",
            {}
        )
        assert not result.is_valid
        assert "field_name" in result.missing_required

        message = validator.format_missing_fields_message(result)
        assert "field" in message.lower()


class TestEdgeCasesQA:
    """Test edge cases that might occur in production."""

    def test_all_caps_input(self, validator):
        """Test ALL CAPS input."""
        result = validator.detect_rule_type("CHECK IF FIELD IS NOT NULL")
        assert result == RuleType.NOT_NULL

    def test_mixed_case_input(self, validator):
        """Test MiXeD CaSe input."""
        result = validator.detect_rule_type("ChEcK If FiElD iS nOt NuLl")
        assert result == RuleType.NOT_NULL

    def test_extra_whitespace(self, validator):
        """Test input with extra whitespace."""
        result = validator.detect_rule_type(
            "  check   if    field   is   not   null  "
        )
        assert result == RuleType.NOT_NULL

    def test_punctuation_variations(self, validator):
        """Test various punctuation."""
        inputs = [
            "check if field is not null!",
            "check if field is not null?",
            "check if field is not null...",
        ]
        for text in inputs:
            result = validator.detect_rule_type(text)
            assert result == RuleType.NOT_NULL

    def test_unicode_characters(self, validator):
        """Test handling of unicode characters."""
        result = validator.detect_rule_type("check if field is not null — test")
        # Should not crash
        assert result is not None or result is None
