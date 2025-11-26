# Data Quality Rules FAQ

## What is a Rule?

A rule is a validation check applied to one or more data fields to ensure data quality. Rules can be simple (check if a field is not null) or complex (validate corporate structure relationships).

## Rule Types

### 1. REGEX Rules
Validate field values against a regular expression pattern.

**Example:**
```csv
R001,Email_Validation,REGEX,email_address,"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",,Invalid email,ERROR,TRUE
```

### 2. NOT_NULL Rules
Ensure a field has a value.

**Example:**
```csv
R002,Required_Field,NOT_NULL,company_name,,,Company name required,ERROR,TRUE
```

### 3. LENGTH Rules
Validate string length (exact, min, or max).

**Example:**
```csv
R003,ISO_Code_Length,LENGTH,country_iso_code,2,2,Must be 2 characters,ERROR,TRUE
```

### 4. RANGE Rules
Validate numeric values are within a range.

**Example:**
```csv
R004,Percentage_Check,RANGE,ownership_percentage,0,100,Must be 0-100,ERROR,TRUE
```

### 5. IN_LIST Rules
Validate value is in an allowed list.

**Example:**
```csv
R005,Status_Check,IN_LIST,status,"Active,Inactive,Pending",,Invalid status,ERROR,TRUE
```

### 6. CUSTOM_FUNCTION Rules
Call a Python function for complex validation.

**Example:**
```csv
R006,BVD_Checksum,CUSTOM_FUNCTION,bvd_id_number,validate_bvd_checksum,,Invalid checksum,ERROR,TRUE
```

## JSON Rule Format

For complex rules, use JSON format:

```json
{
  "type": "composite",
  "operator": "AND",
  "rules": [
    {"field": "city", "validation": "NOT_NULL"},
    {"field": "postcode", "validation": "NOT_NULL"}
  ],
  "error_message": "Complete address required"
}
```

## Common Regex Patterns

| Pattern | Description |
|---------|-------------|
| `^[A-Z]{2}[A-Z0-9]+$` | BvD ID format |
| `^https?://` | URL with protocol |
| `^\d{4}-\d{2}-\d{2}$` | Date YYYY-MM-DD |
| `^[^!]*$` | No exclamation marks |
| `^\+?[0-9\s\-\(\)]+$` | Phone number |

## Severity Levels

- **ERROR**: Critical - record fails validation
- **WARNING**: Non-critical - flagged for review
- **INFO**: Informational only

## Writing Custom Functions

Custom functions must:
1. Accept the field value as first parameter
2. Return a tuple: `(is_valid: bool, error_message: str or None)`

```python
def my_custom_check(value: str) -> Tuple[bool, Optional[str]]:
    if some_condition(value):
        return True, None
    return False, "Validation failed: reason"
```
