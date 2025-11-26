#!/usr/bin/env python3
"""
Generate training data for LoRA fine-tuning from existing rules and documentation.

This script creates JSONL training data in the format:
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

Usage:
    python generate_training_data.py --output training_data.jsonl
"""

import json
import csv
import os
import random
import argparse
from pathlib import Path
from typing import List, Dict, Any


def get_project_root() -> Path:
    """Get project root directory."""
    return Path(__file__).parent.parent


def load_csv_rules(filepath: Path) -> List[Dict]:
    """Load rules from CSV file."""
    rules = []
    if filepath.exists():
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rules.append(row)
    return rules


def load_json_rules(filepath: Path) -> List[Dict]:
    """Load rules from JSON CSV file."""
    rules = []
    if filepath.exists():
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    rule_json = json.loads(row.get('rule_json', '{}'))
                    rules.append({
                        'rule_id': row.get('rule_id'),
                        'rule_name': row.get('rule_name'),
                        'rule_json': rule_json
                    })
                except json.JSONDecodeError:
                    continue
    return rules


def generate_regex_examples(rules: List[Dict]) -> List[Dict]:
    """Generate training examples for REGEX rules."""
    examples = []

    # Templates for user prompts
    prompt_templates = [
        "Create a rule for {field} that checks {description}",
        "I need a validation rule for {field} - {description}",
        "Make a regex rule for {field}: {description}",
        "Validate {field} with pattern {description}",
        "Check if {field} {description}",
        "Create a rule that ensures {field} {description}",
    ]

    for rule in rules:
        if rule.get('rule_type') != 'REGEX':
            continue

        field = rule.get('field_name', '')
        pattern = rule.get('validation_value', '')
        error_msg = rule.get('error_message', '')

        # Generate description from error message
        description = error_msg.lower().replace('invalid', 'is valid').replace('must', 'should')

        for template in random.sample(prompt_templates, min(3, len(prompt_templates))):
            user_prompt = template.format(field=field, description=description)

            assistant_response = f"""I'll create a REGEX rule for {field}.

**Rule Created:**
- Field: `{field}`
- Type: REGEX
- Pattern: `{pattern}`
- Error: {error_msg}

```csv
{rule.get('rule_id')},{rule.get('rule_name')},REGEX,{field},REGEX,"{pattern}",{error_msg},{rule.get('severity', 'ERROR')},TRUE
```"""

            examples.append({
                "messages": [
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": assistant_response}
                ]
            })

    return examples


def generate_not_null_examples(rules: List[Dict]) -> List[Dict]:
    """Generate training examples for NOT_NULL rules."""
    examples = []

    prompt_templates = [
        "Make sure {field} is not empty",
        "{field} should be required",
        "Create a rule that {field} cannot be null",
        "I need {field} to be mandatory",
        "Ensure {field} has a value",
        "{field} must not be blank",
        "gotta have {field}",
        "{field} can't be empty",
    ]

    for rule in rules:
        if rule.get('rule_type') != 'NOT_NULL':
            continue

        field = rule.get('field_name', '')
        error_msg = rule.get('error_message', '')

        for template in random.sample(prompt_templates, min(4, len(prompt_templates))):
            user_prompt = template.format(field=field)

            assistant_response = f"""I'll create a NOT_NULL rule to make {field} required.

**Rule Created:**
- Field: `{field}`
- Type: NOT_NULL
- Error: {error_msg}

```csv
{rule.get('rule_id')},{rule.get('rule_name')},NOT_NULL,{field},NOT_NULL,,{error_msg},{rule.get('severity', 'ERROR')},TRUE
```"""

            examples.append({
                "messages": [
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": assistant_response}
                ]
            })

    return examples


def generate_length_examples(rules: List[Dict]) -> List[Dict]:
    """Generate training examples for LENGTH rules."""
    examples = []

    prompt_templates = [
        "{field} must be exactly {length} characters",
        "Create a length rule for {field}: {min_len} to {max_len} characters",
        "{field} should be {length} chars long",
        "Validate that {field} has length between {min_len} and {max_len}",
    ]

    for rule in rules:
        if rule.get('rule_type') != 'LENGTH':
            continue

        field = rule.get('field_name', '')
        val = rule.get('validation_value', '0,255')
        error_msg = rule.get('error_message', '')

        try:
            parts = val.split(',')
            min_len = parts[0] if parts else '0'
            max_len = parts[1] if len(parts) > 1 else '255'
        except:
            min_len, max_len = '0', '255'

        length = min_len if min_len == max_len else f"{min_len}-{max_len}"

        for template in random.sample(prompt_templates, min(2, len(prompt_templates))):
            user_prompt = template.format(
                field=field, length=length, min_len=min_len, max_len=max_len
            )

            assistant_response = f"""I'll create a LENGTH rule for {field}.

**Rule Created:**
- Field: `{field}`
- Type: LENGTH
- Min: {min_len}, Max: {max_len}
- Error: {error_msg}

```csv
{rule.get('rule_id')},{rule.get('rule_name')},LENGTH,{field},LENGTH,"{min_len},{max_len}",{error_msg},{rule.get('severity', 'ERROR')},TRUE
```"""

            examples.append({
                "messages": [
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": assistant_response}
                ]
            })

    return examples


def generate_range_examples(rules: List[Dict]) -> List[Dict]:
    """Generate training examples for RANGE rules."""
    examples = []

    prompt_templates = [
        "{field} must be between {min_val} and {max_val}",
        "Create a range rule for {field}: {min_val} to {max_val}",
        "Validate {field} is in range {min_val}-{max_val}",
        "{field} should be at least {min_val} and at most {max_val}",
    ]

    for rule in rules:
        if rule.get('rule_type') != 'RANGE':
            continue

        field = rule.get('field_name', '')
        val = rule.get('validation_value', '0,100')
        error_msg = rule.get('error_message', '')

        try:
            parts = val.split(',')
            min_val = parts[0] if parts else '0'
            max_val = parts[1] if len(parts) > 1 else '100'
        except:
            min_val, max_val = '0', '100'

        for template in random.sample(prompt_templates, min(2, len(prompt_templates))):
            user_prompt = template.format(field=field, min_val=min_val, max_val=max_val)

            assistant_response = f"""I'll create a RANGE rule for {field}.

**Rule Created:**
- Field: `{field}`
- Type: RANGE
- Min: {min_val}, Max: {max_val}
- Error: {error_msg}

```csv
{rule.get('rule_id')},{rule.get('rule_name')},RANGE,{field},RANGE,"{min_val},{max_val}",{error_msg},{rule.get('severity', 'ERROR')},TRUE
```"""

            examples.append({
                "messages": [
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": assistant_response}
                ]
            })

    return examples


def generate_in_list_examples(rules: List[Dict]) -> List[Dict]:
    """Generate training examples for IN_LIST rules."""
    examples = []

    prompt_templates = [
        "{field} must be one of: {values}",
        "Create a rule where {field} can only be {values}",
        "Validate {field} is in list: {values}",
        "{field} allowed values are {values}",
    ]

    for rule in rules:
        if rule.get('rule_type') != 'IN_LIST':
            continue

        field = rule.get('field_name', '')
        values = rule.get('validation_value', '')
        error_msg = rule.get('error_message', '')

        for template in random.sample(prompt_templates, min(2, len(prompt_templates))):
            user_prompt = template.format(field=field, values=values)

            assistant_response = f"""I'll create an IN_LIST rule for {field}.

**Rule Created:**
- Field: `{field}`
- Type: IN_LIST
- Allowed: {values}
- Error: {error_msg}

```csv
{rule.get('rule_id')},{rule.get('rule_name')},IN_LIST,{field},IN_LIST,"{values}",{error_msg},{rule.get('severity', 'ERROR')},TRUE
```"""

            examples.append({
                "messages": [
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": assistant_response}
                ]
            })

    return examples


def generate_clarification_examples() -> List[Dict]:
    """Generate examples where the assistant asks for clarification."""
    examples = [
        {
            "messages": [
                {"role": "user", "content": "Create a rule that checks if a field is null"},
                {"role": "assistant", "content": "I can create a NOT_NULL rule, but I need to know which field should be required. Which field should this rule apply to?"}
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "Validate email format"},
                {"role": "assistant", "content": "I can create an email format validation rule. Which email field should I validate? For example: email_address, contact_email, etc."}
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "Check the URL"},
                {"role": "assistant", "content": "I can create a URL validation rule. I need a bit more information:\n\n1. Which URL field should I validate?\n2. What should I check for? (e.g., valid format, no special characters, must start with https)"}
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "Make a length rule"},
                {"role": "assistant", "content": "I can create a LENGTH rule. Please provide:\n\n1. Which field should have length validation?\n2. What are the minimum and maximum lengths allowed?"}
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "Create range validation"},
                {"role": "assistant", "content": "I can create a RANGE rule for numeric validation. I need:\n\n1. Which field should be validated?\n2. What is the minimum value allowed?\n3. What is the maximum value allowed?"}
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "field should be required"},
                {"role": "assistant", "content": "I can create a NOT_NULL rule to make a field required. Which specific field should be mandatory?"}
            ]
        },
    ]
    return examples


def generate_question_answer_examples() -> List[Dict]:
    """Generate Q&A examples about the system."""
    examples = [
        {
            "messages": [
                {"role": "user", "content": "What rule types are supported?"},
                {"role": "assistant", "content": """The system supports these rule types:

1. **REGEX** - Pattern matching using regular expressions
   - Example: Check if email has valid format

2. **NOT_NULL** - Required field validation
   - Example: Ensure postcode is not empty

3. **LENGTH** - String length validation
   - Example: Country code must be exactly 2 characters

4. **RANGE** - Numeric range validation
   - Example: Percentage must be between 0 and 100

5. **IN_LIST** - Allowed values validation
   - Example: Status must be Active, Inactive, or Pending

6. **CUSTOM_FUNCTION** - Python function for complex validation
   - Example: BVD ID checksum validation

For complex rules, we also support JSON formats:
- **Composite** - Multiple conditions with AND/OR
- **Conditional** - If X then validate Y
- **Cross-Field** - Compare two fields"""}
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "How do I create a composite rule?"},
                {"role": "assistant", "content": """Composite rules combine multiple validations with AND/OR logic. Here's how:

**Example**: Address must have city AND country AND postcode

```json
{
  "type": "composite",
  "operator": "AND",
  "rules": [
    {"field": "city", "validation": "NOT_NULL"},
    {"field": "country_iso_code", "validation": "REGEX", "pattern": "^[A-Z]{2}$"},
    {"field": "postcode", "validation": "NOT_NULL"}
  ],
  "error_message": "Complete address required",
  "severity": "ERROR"
}
```

Just describe what you need, like:
- "city AND country AND postcode must all be filled"
- "Either email OR phone must be provided"

I'll create the appropriate JSON rule for you."""}
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "What custom functions are available?"},
                {"role": "assistant", "content": """These custom validation functions are available:

1. **validate_bvd_checksum** - Validates BvD ID format and checksum
   - Checks 2-letter country code prefix
   - Validates alphanumeric format

2. **validate_url_no_special_chars** - Checks URL for disallowed characters
   - Default disallowed: !@#$%^&*

3. **validate_corporate_structure** - Validates ownership hierarchy
   - Checks for circular references
   - Validates maximum ownership depth

4. **validate_date_sequence** - Ensures date1 < date2
   - Useful for start/end date validation

5. **validate_percentage_sum** - Checks percentages sum to 100%
   - Configurable tolerance

6. **validate_email_domain** - Validates against allowed/blocked domains

To use: "Create a custom rule for bvd_id using validate_bvd_checksum"

I can also generate new custom functions if needed!"""}
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "Show me an example of a REGEX rule"},
                {"role": "assistant", "content": """Here's a REGEX rule example:

**Scenario**: Validate email addresses have proper format

**User Request**: "Check if email_address has valid email format"

**Generated Rule**:
```csv
R001,Email_Format_Validation,REGEX,email_address,REGEX,"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",Invalid email format,ERROR,TRUE
```

**Common REGEX patterns**:
- Email: `^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$`
- URL protocol: `^https?://`
- No exclamation points: `^[^!]*$`
- Country code (2 letters): `^[A-Z]{2}$`
- Phone: `^\+?[0-9\s\-\(\)]+$`
- Date YYYY-MM-DD: `^\d{4}-\d{2}-\d{2}$`

Just describe what you need and I'll create the appropriate regex!"""}
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "Can I compare two fields?"},
                {"role": "assistant", "content": """Yes! You can use cross-field validation to compare two fields.

**Example**: Incorporation date must be before dissolution date

```json
{
  "type": "cross_field",
  "validation": "LESS_THAN",
  "field1": "incorporation_date",
  "field2": "dissolution_date",
  "error_message": "Incorporation date must be before dissolution date",
  "severity": "ERROR"
}
```

**Available comparisons**:
- `LESS_THAN` - field1 < field2
- `GREATER_THAN` - field1 > field2
- `EQUALS` - field1 == field2
- `NOT_EQUALS` - field1 != field2

Just say something like:
- "Start date must be before end date"
- "Compare incorporation_date with dissolution_date"

I'll create the cross-field rule for you!"""}
            ]
        },
    ]
    return examples


def generate_typo_examples() -> List[Dict]:
    """Generate examples with typos and informal language."""
    examples = [
        {
            "messages": [
                {"role": "user", "content": "cehck if emial has exclaimation points"},
                {"role": "assistant", "content": """I understand you want to check if email contains exclamation points. I'll create a REGEX rule.

**Rule Created:**
- Field: `email_address`
- Type: REGEX
- Pattern: `^[^!]*$` (no exclamation points)

```csv
R001,Email_No_Exclamation,REGEX,email_address,REGEX,"^[^!]*$",Email should not contain exclamation points,WARNING,TRUE
```"""}
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "make sure the url aint empty yo"},
                {"role": "assistant", "content": """Got it! I'll create a NOT_NULL rule to make the URL field required.

**Rule Created:**
- Field: `web_url`
- Type: NOT_NULL

```csv
R001,URL_Required,NOT_NULL,web_url,NOT_NULL,,URL is required,ERROR,TRUE
```"""}
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "gotta have postcode filled in"},
                {"role": "assistant", "content": """I'll create a NOT_NULL rule to make postcode required.

**Rule Created:**
- Field: `postcode`
- Type: NOT_NULL

```csv
R001,Postcode_Required,NOT_NULL,postcode,NOT_NULL,,Postcode is required,ERROR,TRUE
```"""}
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "countrry code shud be 2 charss"},
                {"role": "assistant", "content": """I'll create a LENGTH rule for country code to be exactly 2 characters.

**Rule Created:**
- Field: `country_iso_code`
- Type: LENGTH
- Min: 2, Max: 2

```csv
R001,Country_Code_Length,LENGTH,country_iso_code,LENGTH,"2,2",Country code must be exactly 2 characters,ERROR,TRUE
```"""}
            ]
        },
    ]
    return examples


def generate_json_rule_examples(json_rules: List[Dict]) -> List[Dict]:
    """Generate training examples for JSON rules."""
    examples = []

    for rule in json_rules:
        rule_json = rule.get('rule_json', {})
        rule_type = rule_json.get('type', '')

        if rule_type == 'composite':
            user_prompt = "Create a composite rule that validates address fields together"
            assistant_response = f"""I'll create a composite rule that validates multiple address fields together.

**Rule Created:**
```json
{json.dumps(rule_json, indent=2)}
```

This rule uses {rule_json.get('operator', 'AND')} logic to combine multiple validations."""

        elif rule_type == 'conditional':
            user_prompt = "Create a conditional rule: if has_email is Y then email must be valid"
            assistant_response = f"""I'll create a conditional rule with if-then logic.

**Rule Created:**
```json
{json.dumps(rule_json, indent=2)}
```

This rule only validates the 'then' field when the 'if' condition is met."""

        elif rule_type == 'cross_field':
            user_prompt = "Create a rule that compares two date fields"
            assistant_response = f"""I'll create a cross-field validation rule.

**Rule Created:**
```json
{json.dumps(rule_json, indent=2)}
```

This rule compares two fields against each other."""
        else:
            continue

        examples.append({
            "messages": [
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": assistant_response}
            ]
        })

    return examples


def main():
    parser = argparse.ArgumentParser(description='Generate training data for LoRA fine-tuning')
    parser.add_argument('--output', '-o', default='training_data.jsonl', help='Output JSONL file')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    random.seed(args.seed)

    project_root = get_project_root()

    # Load existing rules
    csv_rules = load_csv_rules(project_root / 'sample_rules.csv')
    json_rules = load_json_rules(project_root / 'sample_rules_json.csv')

    print(f"Loaded {len(csv_rules)} CSV rules")
    print(f"Loaded {len(json_rules)} JSON rules")

    # Generate examples
    all_examples = []

    # Rule creation examples
    all_examples.extend(generate_regex_examples(csv_rules))
    all_examples.extend(generate_not_null_examples(csv_rules))
    all_examples.extend(generate_length_examples(csv_rules))
    all_examples.extend(generate_range_examples(csv_rules))
    all_examples.extend(generate_in_list_examples(csv_rules))
    all_examples.extend(generate_json_rule_examples(json_rules))

    # Clarification examples
    all_examples.extend(generate_clarification_examples())

    # Q&A examples
    all_examples.extend(generate_question_answer_examples())

    # Typo/informal examples
    all_examples.extend(generate_typo_examples())

    # Shuffle
    random.shuffle(all_examples)

    # Write output
    output_path = project_root / 'training' / args.output
    with open(output_path, 'w') as f:
        for example in all_examples:
            f.write(json.dumps(example) + '\n')

    print(f"\nGenerated {len(all_examples)} training examples")
    print(f"Saved to: {output_path}")

    # Print summary
    print("\nExample breakdown:")
    print(f"  - Rule creation examples: {len(all_examples) - 15}")
    print(f"  - Clarification examples: 6")
    print(f"  - Q&A examples: 5")
    print(f"  - Typo handling examples: 4")


if __name__ == '__main__':
    main()
