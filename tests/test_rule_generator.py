"""
Unit tests for the RuleGenerator module.
Tests rule creation in CSV and JSON formats.
"""

import pytest
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.rule_generator import RuleGenerator, Rule


@pytest.fixture
def generator():
    """Create a RuleGenerator instance with temp output directory."""
    from pathlib import Path
    gen = RuleGenerator()
    gen.output_dir = Path(tempfile.mkdtemp())
    return gen


class TestRuleCreation:
    """Test rule creation methods."""

    def test_create_regex_rule(self, generator):
        """Test REGEX rule creation."""
        rule = generator.create_regex_rule(
            field_name="all_addresses.web_url",
            pattern="^[^!]*$",
            error_message="URL contains exclamation point",
            severity="ERROR"
        )
        assert rule.rule_type == "REGEX"
        assert rule.field_name == "all_addresses.web_url"
        assert rule.validation_value == "^[^!]*$"
        assert rule.severity == "ERROR"

    def test_create_not_null_rule(self, generator):
        """Test NOT_NULL rule creation."""
        rule = generator.create_not_null_rule(
            field_name="acnc.bvd_id_number",
            error_message="BVD ID is required"
        )
        assert rule.rule_type == "NOT_NULL"
        assert rule.field_name == "acnc.bvd_id_number"
        assert "Required" in rule.rule_name

    def test_create_length_rule(self, generator):
        """Test LENGTH rule creation."""
        rule = generator.create_length_rule(
            field_name="acnc.country_iso_code",
            min_length=2,
            max_length=2,
            error_message="Country code must be 2 characters"
        )
        assert rule.rule_type == "LENGTH"
        assert "2,2" in rule.validation_value

    def test_create_range_rule(self, generator):
        """Test RANGE rule creation."""
        rule = generator.create_range_rule(
            field_name="shareholder.percentage",
            min_value=0,
            max_value=100,
            error_message="Percentage must be 0-100"
        )
        assert rule.rule_type == "RANGE"
        assert "0" in rule.validation_value
        assert "100" in rule.validation_value

    def test_create_in_list_rule(self, generator):
        """Test IN_LIST rule creation."""
        rule = generator.create_in_list_rule(
            field_name="company.status",
            allowed_values=["Active", "Inactive", "Pending"],
            error_message="Invalid status"
        )
        assert rule.rule_type == "IN_LIST"
        assert "Active" in rule.validation_value
        assert "Inactive" in rule.validation_value

    def test_create_custom_function_rule(self, generator):
        """Test CUSTOM_FUNCTION rule creation."""
        rule = generator.create_custom_function_rule(
            field_name="acnc.bvd_id_number",
            function_name="validate_bvd_checksum",
            error_message="Invalid BVD checksum"
        )
        assert rule.rule_type == "CUSTOM_FUNCTION"
        assert rule.validation_value == "validate_bvd_checksum"


class TestRuleIdGeneration:
    """Test rule ID generation."""

    def test_unique_ids(self, generator):
        """Test that rule IDs are unique."""
        rule1 = generator.create_regex_rule("field1", "pattern", "error")
        rule2 = generator.create_regex_rule("field2", "pattern", "error")
        assert rule1.rule_id != rule2.rule_id

    def test_id_format(self, generator):
        """Test rule ID format."""
        rule = generator.create_regex_rule("field", "pattern", "error")
        # Should start with 'R' followed by date
        assert rule.rule_id.startswith("R")
        assert "_" in rule.rule_id


class TestRuleConversion:
    """Test rule format conversion."""

    def test_to_csv_row(self, generator):
        """Test CSV row generation."""
        rule = generator.create_regex_rule(
            field_name="test.field",
            pattern="^test$",
            error_message="Test error",
            severity="WARNING"
        )
        csv_row = generator.to_csv_row(rule)
        assert "REGEX" in csv_row
        assert "test.field" in csv_row
        assert "WARNING" in csv_row

    def test_to_json(self, generator):
        """Test JSON conversion."""
        rule = generator.create_regex_rule(
            field_name="test.field",
            pattern="^test$",
            error_message="Test error"
        )
        json_obj = generator.to_json(rule)
        assert json_obj["type"] == "regex"
        assert json_obj["field"] == "test.field"
        assert "validation" in json_obj

    def test_csv_escaping(self, generator):
        """Test CSV escaping for special characters."""
        rule = generator.create_regex_rule(
            field_name="test.field",
            pattern="value1,value2,value3",  # Contains commas
            error_message="Error"
        )
        csv_row = generator.to_csv_row(rule)
        # Pattern with commas should be quoted
        assert '"value1,value2,value3"' in csv_row


class TestRuleSaving:
    """Test saving rules to files."""

    def test_save_rules_csv(self, generator):
        """Test saving rules to CSV file."""
        rules = [
            generator.create_regex_rule("f1", "p1", "e1"),
            generator.create_not_null_rule("f2", "e2"),
        ]
        filepath = generator.save_rules(rules, "test_rules.csv")

        assert os.path.exists(filepath)

        with open(filepath, 'r') as f:
            content = f.read()
            assert "rule_id" in content  # Header
            assert "REGEX" in content
            assert "NOT_NULL" in content

    def test_save_rules_json(self, generator):
        """Test saving rules to JSON file."""
        rules = [
            generator.create_regex_rule("f1", "p1", "e1"),
        ]
        filepath = generator.save_rules_json(rules, "test_rules.json")

        assert os.path.exists(filepath)

        import json
        with open(filepath, 'r') as f:
            data = json.load(f)
            assert len(data) == 1
            assert data[0]["type"] == "regex"

    def test_auto_filename(self, generator):
        """Test auto-generated filename."""
        rules = [generator.create_regex_rule("f", "p", "e")]
        filepath = generator.save_rules(rules)

        assert os.path.exists(filepath)
        assert "rules_" in filepath
        assert ".csv" in filepath


class TestRuleDefaults:
    """Test default values for rules."""

    def test_default_severity(self, generator):
        """Test default severity is ERROR."""
        rule = generator.create_not_null_rule("field", "error")
        assert rule.severity == "ERROR"

    def test_default_is_active(self, generator):
        """Test default is_active is True."""
        rule = generator.create_regex_rule("field", "pattern", "error")
        assert rule.is_active is True
