"""
Rule generation module for creating data quality validation rules.
"""

import csv
import json
import os
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

from .config import config


@dataclass
class Rule:
    """Represents a data quality validation rule."""
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
    """Generates data quality rules in various formats."""

    def __init__(self):
        """Initialize the rule generator."""
        # Ensure output_dir is always a Path object
        output_dir_str = config.rules_output_dir
        if isinstance(output_dir_str, Path):
            self.output_dir = output_dir_str
        else:
            self.output_dir = Path(str(output_dir_str))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._rule_counter = 0

    def _generate_rule_id(self) -> str:
        """Generate a unique rule ID."""
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
        """
        Create a REGEX validation rule.

        Args:
            field_name: The field to validate
            pattern: Regex pattern to match
            error_message: Error message on failure
            severity: ERROR, WARNING, or INFO
            rule_name: Optional custom rule name

        Returns:
            A Rule object
        """
        rule_id = self._generate_rule_id()
        if not rule_name:
            clean_field = field_name.replace('.', '_')
            rule_name = f"Regex_Check_{clean_field}"

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
        clean_field = field_name.replace('.', '_')
        return Rule(
            rule_id=rule_id,
            rule_name=f"Required_{clean_field}",
            rule_type="NOT_NULL",
            field_name=field_name,
            validation_type="NOT_NULL",
            validation_value="",
            error_message=error_message,
            severity=severity
        )

    def create_length_rule(
        self,
        field_name: str,
        min_length: int,
        max_length: int,
        error_message: str,
        severity: str = "ERROR"
    ) -> Rule:
        """Create a LENGTH validation rule."""
        rule_id = self._generate_rule_id()
        clean_field = field_name.replace('.', '_')
        return Rule(
            rule_id=rule_id,
            rule_name=f"Length_Check_{clean_field}",
            rule_type="LENGTH",
            field_name=field_name,
            validation_type="LENGTH",
            validation_value=f"{min_length},{max_length}",
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
        clean_field = field_name.replace('.', '_')
        return Rule(
            rule_id=rule_id,
            rule_name=f"Range_Check_{clean_field}",
            rule_type="RANGE",
            field_name=field_name,
            validation_type="RANGE",
            validation_value=f"{min_value},{max_value}",
            error_message=error_message,
            severity=severity
        )

    def create_in_list_rule(
        self,
        field_name: str,
        allowed_values: List[str],
        error_message: str,
        severity: str = "ERROR"
    ) -> Rule:
        """Create an IN_LIST validation rule."""
        rule_id = self._generate_rule_id()
        clean_field = field_name.replace('.', '_')
        return Rule(
            rule_id=rule_id,
            rule_name=f"InList_Check_{clean_field}",
            rule_type="IN_LIST",
            field_name=field_name,
            validation_type="IN_LIST",
            validation_value=",".join(allowed_values),
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
        """Convert a rule to a CSV row string."""
        val = rule.validation_value
        if ',' in val or '"' in val:
            val = f'"{val.replace(chr(34), chr(34)+chr(34))}"'

        return ",".join([
            rule.rule_id,
            rule.rule_name,
            rule.rule_type,
            rule.field_name,
            rule.validation_type,
            val,
            rule.error_message,
            rule.severity,
            "TRUE" if rule.is_active else "FALSE"
        ])

    def to_json(self, rule: Rule) -> Dict[str, Any]:
        """Convert a rule to JSON format."""
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

    def save_rules(
        self,
        rules: List[Rule],
        filename: Optional[str] = None
    ) -> str:
        """
        Save rules to a CSV file.

        Args:
            rules: List of Rule objects to save
            filename: Optional filename (auto-generated if not provided)

        Returns:
            Path to the saved file
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"rules_{timestamp}.csv"

        filepath = self.output_dir / filename

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

        return str(filepath)

    def save_rules_json(
        self,
        rules: List[Rule],
        filename: Optional[str] = None
    ) -> str:
        """
        Save rules to a JSON file.

        Args:
            rules: List of Rule objects to save
            filename: Optional filename

        Returns:
            Path to the saved file
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"rules_{timestamp}.json"

        filepath = self.output_dir / filename

        rules_json = [self.to_json(rule) for rule in rules]

        with open(filepath, 'w') as f:
            json.dump(rules_json, f, indent=2)

        return str(filepath)
