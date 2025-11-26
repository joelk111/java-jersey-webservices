"""
Code generation module for creating custom validation functions.
"""

from typing import List, Optional
from dataclasses import dataclass
import os
from pathlib import Path

from .config import config


@dataclass
class CustomFunction:
    """Represents a generated custom validation function."""
    name: str
    description: str
    parameters: List[str]
    code: str


class CodeGenerator:
    """Generates Python code for custom validation functions."""

    FUNCTION_TEMPLATE = '''def {name}({params}) -> Tuple[bool, Optional[str]]:
    """
    {description}

    Args:
{args_doc}

    Returns:
        Tuple of (is_valid, error_message)
    """
{body}
'''

    IMPORTS = '''"""
Auto-generated custom validation function.
"""
from typing import Tuple, Optional
import re

'''

    def generate_function(
        self,
        name: str,
        description: str,
        validation_logic: str,
        parameters: Optional[List[str]] = None
    ) -> CustomFunction:
        """
        Generate a custom validation function.

        Args:
            name: Function name (snake_case)
            description: What the function validates
            validation_logic: The actual validation code
            parameters: Additional parameters beyond 'value'

        Returns:
            A CustomFunction object
        """
        params = ["value: str"]
        if parameters:
            params.extend(parameters)
        params_str = ", ".join(params)

        # Generate args documentation
        args_lines = ["        value: The field value to validate"]
        if parameters:
            for param in parameters:
                param_name = param.split(":")[0].strip() if ":" in param else param
                args_lines.append(f"        {param_name}: Additional parameter")
        args_doc = "\n".join(args_lines)

        # Generate function body with proper indentation
        body_lines = []
        for line in validation_logic.strip().split("\n"):
            body_lines.append(f"    {line}")
        body = "\n".join(body_lines)

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
        # Escape backslashes for the pattern string
        escaped_pattern = pattern.replace("\\", "\\\\")

        validation_logic = f'''if not value:
    return True, None  # Empty handled by NOT_NULL

pattern = r"{escaped_pattern}"
if not re.match(pattern, value):
    return False, "{error_message}"
return True, None'''

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
        validation_logic = f'''if not value:
    return True, None  # Empty handled by NOT_NULL

disallowed = "{disallowed_chars}"
found = [c for c in disallowed if c in value]
if found:
    return False, "{error_message}: found " + ", ".join(repr(c) for c in found)
return True, None'''

        return self.generate_function(
            name=name,
            description=f"Checks that value does not contain: {disallowed_chars}",
            validation_logic=validation_logic
        )

    def generate_url_validation_function(
        self,
        name: str,
        require_https: bool = False
    ) -> CustomFunction:
        """Generate URL validation function."""
        protocol = "https" if require_https else "https?"

        validation_logic = f'''if not value:
    return True, None

import re
pattern = r"^{protocol}://[a-zA-Z0-9][a-zA-Z0-9-]*[a-zA-Z0-9]*(\\.[a-zA-Z]{{2,}})+(/.*)?$"
if not re.match(pattern, value, re.IGNORECASE):
    return False, "Invalid URL format"
return True, None'''

        return self.generate_function(
            name=name,
            description=f"Validates URL format (require_https={require_https})",
            validation_logic=validation_logic
        )

    def generate_date_comparison_function(
        self,
        name: str,
        comparison: str = "before"  # "before" or "after"
    ) -> CustomFunction:
        """Generate date comparison function."""
        operator = "<" if comparison == "before" else ">"
        comparison_word = comparison

        validation_logic = f'''from datetime import datetime

if not value:
    return True, None

date_format = "%Y-%m-%d"
try:
    # value should be "date1,date2" format
    parts = value.split(",")
    if len(parts) != 2:
        return False, "Expected format: date1,date2"

    date1 = datetime.strptime(parts[0].strip(), date_format)
    date2 = datetime.strptime(parts[1].strip(), date_format)

    if not (date1 {operator} date2):
        return False, f"First date must be {comparison_word} second date"

    return True, None
except ValueError as e:
    return False, f"Invalid date format: {{e}}"'''

        return self.generate_function(
            name=name,
            description=f"Validates that first date is {comparison} second date",
            validation_logic=validation_logic
        )

    def save_function(
        self,
        func: CustomFunction,
        directory: Optional[str] = None
    ) -> str:
        """
        Save a function to a Python file.

        Args:
            func: The CustomFunction to save
            directory: Output directory (defaults to custom_functions/)

        Returns:
            Path to the saved file
        """
        if not directory:
            base_dir = Path(config.rules_output_dir).parent
            directory = base_dir / "custom_functions"
        else:
            directory = Path(directory)

        directory.mkdir(parents=True, exist_ok=True)

        filepath = directory / f"{func.name}.py"

        with open(filepath, 'w') as f:
            f.write(self.IMPORTS)
            f.write(func.code)
            f.write("\n")

        return str(filepath)

    def get_full_code(self, func: CustomFunction) -> str:
        """Get the complete code including imports."""
        return self.IMPORTS + func.code
