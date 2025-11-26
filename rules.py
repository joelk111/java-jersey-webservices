"""
Sample Custom Validation Functions for ORBIS Data Quality Rules Engine

This module contains example custom validation functions that can be called
from the rules engine when standard validation types are not sufficient.

Each function should:
1. Accept the field value as the first parameter
2. Accept optional additional parameters as needed
3. Return a tuple of (is_valid: bool, error_message: str or None)
"""

import re
from typing import Tuple, Optional, Any, Dict, List
from datetime import datetime


def validate_bvd_checksum(value: str) -> Tuple[bool, Optional[str]]:
    """
    Validates the BvD ID checksum using a custom algorithm.
    BvD IDs start with a 2-letter country code followed by alphanumeric characters.

    Args:
        value: The BvD ID to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not value or len(value) < 3:
        return False, "BvD ID too short"

    # Check country code prefix (first 2 chars must be uppercase letters)
    country_code = value[:2]
    if not country_code.isalpha() or not country_code.isupper():
        return False, f"Invalid country code prefix: {country_code}"

    # Validate the rest contains only alphanumeric
    remainder = value[2:]
    if not remainder.isalnum():
        return False, "BvD ID contains invalid characters"

    # Example checksum validation (simplified)
    # In production, this would use the actual BvD checksum algorithm
    return True, None


def validate_url_no_special_chars(value: str, disallowed: str = "!@#$%^&*") -> Tuple[bool, Optional[str]]:
    """
    Validates that a URL does not contain specified special characters.

    Args:
        value: The URL to validate
        disallowed: String of disallowed characters (default: "!@#$%^&*")

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not value:
        return True, None  # Empty values pass (use NOT_NULL for required check)

    found_chars = [c for c in disallowed if c in value]
    if found_chars:
        return False, f"URL contains disallowed characters: {', '.join(found_chars)}"

    return True, None


def validate_corporate_structure(
    records: List[Dict[str, Any]],
    max_depth: int = 10,
    check_circular: bool = True
) -> Tuple[bool, Optional[str]]:
    """
    Validates corporate ownership structure for circular references and depth limits.

    Args:
        records: List of ownership records with bvd_id_number and shareholder_bvd_id_number
        max_depth: Maximum allowed ownership depth
        check_circular: Whether to check for circular ownership

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Build ownership graph
    ownership_graph = {}
    for record in records:
        company_id = record.get('bvd_id_number')
        shareholder_id = record.get('shareholder_bvd_id_number')

        if company_id and shareholder_id:
            if company_id not in ownership_graph:
                ownership_graph[company_id] = []
            ownership_graph[company_id].append(shareholder_id)

    # Check for circular references
    if check_circular:
        visited = set()
        rec_stack = set()

        def has_cycle(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)

            for neighbor in ownership_graph.get(node, []):
                if neighbor not in visited:
                    if has_cycle(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True

            rec_stack.remove(node)
            return False

        for node in ownership_graph:
            if node not in visited:
                if has_cycle(node):
                    return False, "Circular ownership reference detected"

    # Check ownership depth
    def get_depth(node: str, current_depth: int = 0) -> int:
        if current_depth >= max_depth:
            return current_depth
        max_child_depth = current_depth
        for child in ownership_graph.get(node, []):
            child_depth = get_depth(child, current_depth + 1)
            max_child_depth = max(max_child_depth, child_depth)
        return max_child_depth

    for node in ownership_graph:
        depth = get_depth(node)
        if depth >= max_depth:
            return False, f"Ownership depth ({depth}) exceeds maximum allowed ({max_depth})"

    return True, None


def validate_date_sequence(
    date1_str: str,
    date2_str: str,
    date_format: str = "%Y-%m-%d"
) -> Tuple[bool, Optional[str]]:
    """
    Validates that date1 is before date2.

    Args:
        date1_str: First date string
        date2_str: Second date string
        date_format: Date format string (default: YYYY-MM-DD)

    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        date1 = datetime.strptime(date1_str, date_format)
        date2 = datetime.strptime(date2_str, date_format)

        if date1 >= date2:
            return False, f"Date {date1_str} must be before {date2_str}"

        return True, None
    except ValueError as e:
        return False, f"Invalid date format: {str(e)}"


def validate_percentage_sum(
    values: List[float],
    expected_sum: float = 100.0,
    tolerance: float = 0.01
) -> Tuple[bool, Optional[str]]:
    """
    Validates that a list of percentages sums to an expected value.

    Args:
        values: List of percentage values
        expected_sum: Expected sum (default: 100.0)
        tolerance: Allowed deviation (default: 0.01)

    Returns:
        Tuple of (is_valid, error_message)
    """
    actual_sum = sum(values)
    if abs(actual_sum - expected_sum) > tolerance:
        return False, f"Percentages sum to {actual_sum}, expected {expected_sum}"

    return True, None


def validate_email_domain(
    email: str,
    allowed_domains: Optional[List[str]] = None,
    blocked_domains: Optional[List[str]] = None
) -> Tuple[bool, Optional[str]]:
    """
    Validates email domain against allowed/blocked lists.

    Args:
        email: Email address to validate
        allowed_domains: List of allowed domains (if set, only these are allowed)
        blocked_domains: List of blocked domains

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not email or '@' not in email:
        return False, "Invalid email format"

    domain = email.split('@')[1].lower()

    if blocked_domains and domain in [d.lower() for d in blocked_domains]:
        return False, f"Email domain {domain} is blocked"

    if allowed_domains:
        if domain not in [d.lower() for d in allowed_domains]:
            return False, f"Email domain {domain} is not in allowed list"

    return True, None


def validate_regex_custom(value: str, pattern: str, flags: int = 0) -> Tuple[bool, Optional[str]]:
    """
    Custom regex validation with optional flags.

    Args:
        value: Value to validate
        pattern: Regex pattern
        flags: Regex flags (e.g., re.IGNORECASE)

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not value:
        return True, None

    try:
        if re.match(pattern, value, flags):
            return True, None
        else:
            return False, f"Value does not match pattern: {pattern}"
    except re.error as e:
        return False, f"Invalid regex pattern: {str(e)}"


# Registry of all custom functions (for dynamic lookup)
CUSTOM_FUNCTIONS = {
    'validate_bvd_checksum': validate_bvd_checksum,
    'validate_url_no_special_chars': validate_url_no_special_chars,
    'validate_corporate_structure': validate_corporate_structure,
    'validate_date_sequence': validate_date_sequence,
    'validate_percentage_sum': validate_percentage_sum,
    'validate_email_domain': validate_email_domain,
    'validate_regex_custom': validate_regex_custom,
}


def get_custom_function(name: str):
    """
    Retrieves a custom function by name from the registry.

    Args:
        name: Name of the function

    Returns:
        The function if found, None otherwise
    """
    return CUSTOM_FUNCTIONS.get(name)


def register_custom_function(name: str, func):
    """
    Registers a new custom function.

    Args:
        name: Name to register the function under
        func: The function to register
    """
    CUSTOM_FUNCTIONS[name] = func
