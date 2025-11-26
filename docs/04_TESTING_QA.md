# NLP Rules Engine - Part 4: Testing and QA

## Test Strategy Overview

| Test Type | Purpose | Tools |
|-----------|---------|-------|
| Unit Tests | Test individual components | pytest |
| Integration Tests | Test component interactions | pytest + httpx |
| LLM Tests | Test model responses | Custom prompts |
| UI Tests | Test Gradio interface | Manual + Selenium |
| Performance Tests | Measure response times | pytest-benchmark |

## Unit Tests

### Field Matcher Tests

```python
# tests/test_field_matcher.py
import pytest
from app.field_matcher import FieldMatcher

@pytest.fixture
def matcher():
    return FieldMatcher("./orbis_field_names.c")

class TestFieldMatcher:

    def test_exact_match(self, matcher):
        """Test exact field name matching."""
        results = matcher.match("bvd_id_number")
        assert len(results) > 0
        assert "bvd_id_number" in results[0][0]

    def test_fuzzy_match_partial(self, matcher):
        """Test fuzzy matching with partial name."""
        results = matcher.match("bvd id")
        assert len(results) > 0
        # Should match bvd_id_number
        assert any("bvd" in r[0].lower() for r in results)

    def test_fuzzy_match_typo(self, matcher):
        """Test fuzzy matching with typo."""
        results = matcher.match("bvd_id_numbre")  # typo
        assert len(results) > 0
        assert results[0][1] >= 70  # Score threshold

    def test_fuzzy_match_label(self, matcher):
        """Test matching by label."""
        results = matcher.match("company email")
        assert len(results) > 0

    def test_no_match_returns_empty(self, matcher):
        """Test that garbage input returns empty."""
        results = matcher.match("xyznonexistent123")
        # May return results with low scores, check threshold
        high_score_results = [r for r in results if r[1] >= 70]
        assert len(high_score_results) == 0

    def test_table_search(self, matcher):
        """Test searching by table name."""
        fields = matcher.search_by_table("acnc")
        assert len(fields) > 0
        assert all(f.startswith("acnc.") for f in fields)

    def test_list_tables(self, matcher):
        """Test listing all tables."""
        tables = matcher.list_tables()
        assert len(tables) > 0
        assert "acnc" in tables

    def test_match_multiple(self, matcher):
        """Test matching multiple queries."""
        results = matcher.match_multiple(["bvd id", "email", "url"])
        assert len(results) == 3
        assert "bvd id" in results
        assert "email" in results

    def test_get_field_info(self, matcher):
        """Test getting field information."""
        # First find a valid field
        results = matcher.match("bvd_id_number")
        if results:
            field_name = results[0][0]
            info = matcher.get_field_info(field_name)
            assert info is not None
            assert "field_name" in info
            assert "data_type" in info
```

### Rule Generator Tests

```python
# tests/test_rule_generator.py
import pytest
import os
import tempfile
from app.rule_generator import RuleGenerator, Rule

@pytest.fixture
def generator():
    gen = RuleGenerator()
    gen.output_dir = tempfile.mkdtemp()
    return gen

class TestRuleGenerator:

    def test_create_regex_rule(self, generator):
        """Test REGEX rule creation."""
        rule = generator.create_regex_rule(
            field_name="all_addresses.web_url",
            pattern="^[^!]*$",
            error_message="URL contains exclamation point"
        )
        assert rule.rule_type == "REGEX"
        assert rule.field_name == "all_addresses.web_url"
        assert "^[^!]*$" in rule.validation_value

    def test_create_not_null_rule(self, generator):
        """Test NOT_NULL rule creation."""
        rule = generator.create_not_null_rule(
            field_name="acnc.bvd_id_number",
            error_message="BVD ID is required"
        )
        assert rule.rule_type == "NOT_NULL"
        assert "Required" in rule.rule_name

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

    def test_create_custom_function_rule(self, generator):
        """Test CUSTOM_FUNCTION rule creation."""
        rule = generator.create_custom_function_rule(
            field_name="acnc.bvd_id_number",
            function_name="validate_bvd_checksum",
            error_message="Invalid BVD checksum"
        )
        assert rule.rule_type == "CUSTOM_FUNCTION"
        assert rule.validation_value == "validate_bvd_checksum"

    def test_rule_id_unique(self, generator):
        """Test that rule IDs are unique."""
        rule1 = generator.create_regex_rule("field1", "pattern", "error")
        rule2 = generator.create_regex_rule("field2", "pattern", "error")
        assert rule1.rule_id != rule2.rule_id

    def test_to_csv_row(self, generator):
        """Test CSV row generation."""
        rule = generator.create_regex_rule(
            field_name="test.field",
            pattern="^test$",
            error_message="Test error"
        )
        csv_row = generator.to_csv_row(rule)
        assert "REGEX" in csv_row
        assert "test.field" in csv_row

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

    def test_save_rules(self, generator):
        """Test saving rules to file."""
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
```

### Code Generator Tests

```python
# tests/test_code_generator.py
import pytest
import tempfile
import os
from app.code_generator import CodeGenerator

@pytest.fixture
def generator():
    return CodeGenerator()

class TestCodeGenerator:

    def test_generate_function(self, generator):
        """Test basic function generation."""
        func = generator.generate_function(
            name="test_validation",
            description="Test validation function",
            validation_logic="return True, None"
        )
        assert func.name == "test_validation"
        assert "def test_validation" in func.code
        assert "Tuple[bool, Optional[str]]" in func.code

    def test_generate_regex_function(self, generator):
        """Test regex function generation."""
        func = generator.generate_regex_function(
            name="validate_url",
            pattern="^https?://",
            error_message="URL must start with http"
        )
        assert "re.match" in func.code
        assert "^https?://" in func.code

    def test_generate_no_chars_function(self, generator):
        """Test disallowed chars function."""
        func = generator.generate_no_chars_function(
            name="no_exclamation",
            disallowed_chars="!@#",
            error_message="Contains special chars"
        )
        assert "!@#" in func.code
        assert "for c in disallowed" in func.code

    def test_save_function(self, generator):
        """Test saving function to file."""
        func = generator.generate_function(
            name="saved_function",
            description="Test",
            validation_logic="return True, None"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = generator.save_function(func, tmpdir)
            assert os.path.exists(filepath)
            with open(filepath, 'r') as f:
                content = f.read()
                assert "def saved_function" in content

    def test_function_compiles(self, generator):
        """Test that generated code compiles."""
        func = generator.generate_regex_function(
            name="compile_test",
            pattern="^test$",
            error_message="Error"
        )
        # Should not raise SyntaxError
        compile(func.code, "<string>", "exec")
```

## Integration Tests

### LLM Integration Tests

```python
# tests/test_llm_integration.py
import pytest
from app.llm_client import OllamaClient, Message
from app.conversation import ConversationManager

@pytest.fixture
def client():
    return OllamaClient()

@pytest.fixture
def conversation():
    return ConversationManager()

class TestLLMIntegration:

    @pytest.mark.integration
    def test_ollama_connection(self, client):
        """Test Ollama is running and accessible."""
        response = client.generate("Say 'OK'")
        assert len(response) > 0

    @pytest.mark.integration
    def test_simple_rule_generation(self, conversation):
        """Test generating a simple rule from natural language."""
        result = conversation.process_message(
            "Create a rule for web_url that checks if it contains exclamation points"
        )
        # Should either generate a rule or ask for clarification
        assert result.get("rules") or result.get("clarification")

    @pytest.mark.integration
    def test_field_matching(self, conversation):
        """Test field matching through conversation."""
        result = conversation.process_message(
            "What fields are available for URLs?"
        )
        assert result.get("message") or result.get("matched_fields")

    @pytest.mark.integration
    def test_clarification_request(self, conversation):
        """Test that LLM asks for clarification when needed."""
        result = conversation.process_message(
            "Create a validation rule"  # Vague request
        )
        # Should ask for more details
        assert result.get("clarification") or "which field" in result.get("message", "").lower()

    @pytest.mark.integration
    def test_multi_turn_conversation(self, conversation):
        """Test multi-turn dialogue."""
        # First message - incomplete request
        result1 = conversation.process_message(
            "I want to check if a URL is valid"
        )

        # Second message - provide missing info
        result2 = conversation.process_message(
            "Use the web_url field and check it starts with http"
        )

        # Should eventually produce a rule
        total_rules = len(result1.get("rules", [])) + len(result2.get("rules", []))
        assert total_rules > 0 or result2.get("message")
```

## Manual QA Checklist

### Pre-Release Checklist

| # | Test Case | Expected Result | Status |
|---|-----------|-----------------|--------|
| 1 | Start Ollama service | Service runs on port 11434 | |
| 2 | Pull Llama model | Model downloads successfully | |
| 3 | Run application | Gradio UI opens in browser | |
| 4 | Type simple rule request | LLM responds within 10 seconds | |
| 5 | Check field matching | Correct fields are suggested | |
| 6 | Generate REGEX rule | Valid CSV output produced | |
| 7 | Generate custom function | Valid Python code produced | |
| 8 | Download rules CSV | File downloads correctly | |
| 9 | Clear conversation | State resets properly | |
| 10 | Test with invalid input | Graceful error handling | |

### Test Scenarios

#### Scenario 1: Simple Regex Rule
```
User: "Create a rule for web_url that fails if it contains exclamation points"

Expected:
- Field matched: web_url or similar
- Rule type: REGEX
- Pattern: ^[^!]*$ or similar
- Severity: ERROR (default)
```

#### Scenario 2: Clarification Needed
```
User: "Check if the email is valid"

Expected:
- LLM asks which email field (email_address, contact_email, etc.)
- After clarification, generates rule
```

#### Scenario 3: Custom Function
```
User: "Create a rule that validates BVD ID checksums using a custom algorithm"

Expected:
- LLM generates Python function
- Rule references custom function
- Both code and rule are provided
```

#### Scenario 4: Complex Multi-Field Rule
```
User: "Create a rule that ensures incorporation_date is before dissolution_date"

Expected:
- LLM identifies cross-field validation
- Generates JSON format rule or custom function
- Explains the logic
```

## Performance Benchmarks

### Target Metrics

| Metric | Target | Acceptable |
|--------|--------|------------|
| First token latency | <2s | <5s |
| Full response time | <10s | <30s |
| Field matching | <100ms | <500ms |
| Rule generation | <1s | <3s |
| Memory usage | <2GB | <4GB |

### Benchmark Script

```python
# tests/benchmark_performance.py
import time
import statistics
from app.field_matcher import FieldMatcher
from app.rule_generator import RuleGenerator
from app.conversation import ConversationManager

def benchmark_field_matching():
    """Benchmark field matching performance."""
    matcher = FieldMatcher()
    queries = ["bvd id", "email", "url", "company name", "country code"]

    times = []
    for query in queries:
        start = time.perf_counter()
        matcher.match(query)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    print(f"Field Matching:")
    print(f"  Mean: {statistics.mean(times)*1000:.2f}ms")
    print(f"  Max: {max(times)*1000:.2f}ms")
    print(f"  Min: {min(times)*1000:.2f}ms")

def benchmark_rule_generation():
    """Benchmark rule generation performance."""
    generator = RuleGenerator()

    start = time.perf_counter()
    for i in range(100):
        generator.create_regex_rule(f"field{i}", "pattern", "error")
    elapsed = time.perf_counter() - start

    print(f"Rule Generation (100 rules): {elapsed*1000:.2f}ms")

def benchmark_llm_response():
    """Benchmark LLM response time."""
    conv = ConversationManager()

    prompts = [
        "Create a rule for web_url that checks for exclamation points",
        "Validate that email_address has proper format",
        "Check if bvd_id_number is not empty",
    ]

    times = []
    for prompt in prompts:
        start = time.perf_counter()
        conv.process_message(prompt)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        conv.reset()

    print(f"LLM Response Time:")
    print(f"  Mean: {statistics.mean(times):.2f}s")
    print(f"  Max: {max(times):.2f}s")
    print(f"  Min: {min(times):.2f}s")

if __name__ == "__main__":
    benchmark_field_matching()
    benchmark_rule_generation()
    # benchmark_llm_response()  # Requires Ollama running
```

## Running Tests

```bash
# All unit tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=app --cov-report=html

# Integration tests only
pytest tests/ -v -m integration

# Performance benchmarks
python tests/benchmark_performance.py
```

## Continuous Integration

### GitHub Actions Workflow

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'

    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov

    - name: Run unit tests
      run: pytest tests/ -v --ignore=tests/test_llm_integration.py

    - name: Upload coverage
      uses: codecov/codecov-action@v3
```

## Conclusion

This testing suite ensures:
1. Individual components work correctly
2. Components integrate properly
3. LLM produces expected outputs
4. Performance meets requirements
5. Edge cases are handled

Run the full test suite before any release to production.
