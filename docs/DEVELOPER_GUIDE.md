# NLP Rules Engine - Developer Guide

## Table of Contents

1. [Quick Start](#quick-start)
2. [Architecture Overview](#architecture-overview)
3. [Project Structure](#project-structure)
4. [Configuration](#configuration)
5. [Running Locally](#running-locally)
6. [Docker Deployment](#docker-deployment)
7. [Adding Custom Validation Functions](#adding-custom-validation-functions)
8. [Testing](#testing)
9. [Troubleshooting](#troubleshooting)

## Quick Start

```bash
# 1. Clone repository
git clone https://github.com/joelk111/java-jersey-webservices.git
cd java-jersey-webservices

# 2. Run setup script
./scripts/setup.sh

# 3. Start Ollama and pull model
ollama serve &
ollama pull llama3.1:8b-instruct-q4_0

# 4. Run the application
source venv/bin/activate
python -m app.main

# Open http://localhost:7860 in your browser
```

## Architecture Overview

```
┌────────────────────────────────────────────────────────────────────┐
│                         Gradio Web UI                              │
│                        (app/main.py)                               │
└────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌────────────────────────────────────────────────────────────────────┐
│                    Conversation Manager                            │
│                    (app/conversation.py)                           │
│                                                                    │
│  ┌─────────────────┐  ┌──────────────────┐  ┌─────────────────┐   │
│  │ Pre-Validation  │  │  LLM Processing  │  │   Fallback      │   │
│  │ (Local Check)   │  │  (Tool Calling)  │  │   Processing    │   │
│  └─────────────────┘  └──────────────────┘  └─────────────────┘   │
└────────────────────────────────────────────────────────────────────┘
                                │
          ┌─────────────────────┼─────────────────────┐
          ▼                     ▼                     ▼
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│  Field Matcher   │  │  Rule Generator  │  │  Code Generator  │
│  (field_matcher) │  │ (rule_generator) │  │ (code_generator) │
└──────────────────┘  └──────────────────┘  └──────────────────┘
          │                     │                     │
          ▼                     ▼                     ▼
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│ orbis_field_     │  │ generated_rules/ │  │ custom_functions/│
│ names.c          │  │ *.csv            │  │ *.py             │
└──────────────────┘  └──────────────────┘  └──────────────────┘
```

## Project Structure

```
nlp-rules-engine/
├── app/                          # Application source code
│   ├── __init__.py
│   ├── main.py                   # Entry point & Gradio UI
│   ├── config.py                 # Environment configuration
│   ├── llm_client.py             # Ollama LLM interface
│   ├── field_matcher.py          # Fuzzy field matching
│   ├── rule_generator.py         # CSV/JSON rule generation
│   ├── code_generator.py         # Python function generation
│   ├── conversation.py           # Multi-turn dialogue
│   └── requirements_validator.py # Required field detection
│
├── config/                       # Environment configurations
│   ├── development.env
│   ├── staging.env
│   └── production.env
│
├── tests/                        # Test suite
│   ├── test_field_matcher.py
│   ├── test_rule_generator.py
│   ├── test_requirements_validator.py
│   └── test_qa_imperfect_prompts.py
│
├── scripts/                      # Utility scripts
│   ├── setup.sh
│   ├── run.sh
│   └── test.sh
│
├── docs/                         # Documentation
│   ├── 01_SYSTEM_OVERVIEW.md
│   ├── 02_INSTALLATION_SETUP.md
│   ├── 03_DEVELOPMENT_GUIDE.md
│   ├── 04_TESTING_QA.md
│   └── DEVELOPER_GUIDE.md
│
├── Dockerfile                    # Production Docker image
├── Dockerfile.dev                # Development Docker image
├── docker-compose.yml            # Production compose
├── docker-compose.dev.yml        # Development compose
├── requirements.txt              # Python dependencies
├── orbis_field_names.c           # Field dictionary
├── sample_rules.csv              # Example rules
└── rules.py                      # Custom functions
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ENVIRONMENT` | Environment name | `development` |
| `OLLAMA_HOST` | Ollama API URL | `http://localhost:11434` |
| `OLLAMA_MODEL` | LLM model to use | `llama3.1:8b-instruct-q4_0` |
| `APP_HOST` | Application host | `0.0.0.0` |
| `APP_PORT` | Application port | `7860` |
| `DEBUG` | Enable debug mode | `false` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `FUZZY_THRESHOLD` | Field match threshold | `70` |
| `LLM_TEMPERATURE` | LLM temperature | `0.7` |
| `LLM_TIMEOUT` | LLM timeout (seconds) | `120` |

### Switching Environments

```bash
# Development
export ENVIRONMENT=development
python -m app.main

# Production
export ENVIRONMENT=production
python -m app.main

# Or use the run script
./scripts/run.sh development
./scripts/run.sh production
```

## Running Locally

### Prerequisites

- Python 3.10+
- Ollama installed and running
- 16GB RAM (minimum)
- GPU with 8GB VRAM (recommended)

### Step-by-Step

```bash
# 1. Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 2. Start Ollama
ollama serve

# 3. Pull the model (in another terminal)
ollama pull llama3.1:8b-instruct-q4_0

# 4. Setup Python environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 5. Run application
python -m app.main
```

## Docker Deployment

### Development Mode

```bash
# Start with hot-reloading
docker-compose -f docker-compose.dev.yml up

# Rebuild after changes
docker-compose -f docker-compose.dev.yml up --build
```

### Production Mode

```bash
# Build and start
docker-compose up -d

# View logs
docker-compose logs -f nlp-rules-engine

# Stop
docker-compose down
```

### First-Time Setup

The `ollama-init` service automatically pulls the model on first run:

```bash
docker-compose up -d
# Wait for model download (5-10 minutes)
docker-compose logs ollama-init
```

## Adding Custom Validation Functions

### 1. Create Function in rules.py

```python
# rules.py

def validate_my_custom_check(value: str, param1: str = None) -> Tuple[bool, Optional[str]]:
    """
    My custom validation.

    Args:
        value: The field value to validate
        param1: Optional parameter

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not value:
        return True, None

    # Your validation logic here
    if some_condition(value):
        return True, None

    return False, "Validation failed: reason"

# Register in CUSTOM_FUNCTIONS dict
CUSTOM_FUNCTIONS['validate_my_custom_check'] = validate_my_custom_check
```

### 2. Use in Rules

```csv
R001,My_Custom_Rule,CUSTOM_FUNCTION,my_field,CUSTOM_FUNCTION,validate_my_custom_check,Custom validation failed,ERROR,TRUE
```

### 3. LLM Can Generate Functions

When a user requests a validation that doesn't exist, the system generates Python code:

```
User: "Create a rule that validates BVD IDs using the Luhn algorithm"

System generates:
- Rule CSV referencing function
- Python code for the function
```

## Testing

### Run All Tests

```bash
./scripts/test.sh
# Or
python -m pytest tests/ -v
```

### Run Specific Tests

```bash
# Field matcher tests
python -m pytest tests/test_field_matcher.py -v

# Rule generator tests
python -m pytest tests/test_rule_generator.py -v

# QA tests with imperfect prompts
python -m pytest tests/test_qa_imperfect_prompts.py -v
```

### Coverage Report

```bash
python -m pytest tests/ --cov=app --cov-report=html
# Open htmlcov/index.html
```

## Troubleshooting

### Ollama Connection Failed

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama
ollama serve

# Check model is available
ollama list
```

### Model Not Responding

```bash
# Check GPU memory
nvidia-smi

# Use smaller model
export OLLAMA_MODEL=llama3.2:3b-instruct-q4_0
```

### Field Not Found

1. Check `orbis_field_names.c` contains the field
2. Try different search terms
3. Lower `FUZZY_THRESHOLD` for more matches

### Docker Issues

```bash
# Rebuild images
docker-compose down
docker-compose build --no-cache
docker-compose up

# Check container logs
docker-compose logs -f

# Reset volumes
docker-compose down -v
```

## API Reference

### ConversationManager

```python
from app.conversation import ConversationManager

# Initialize
manager = ConversationManager()

# Process user message
result = manager.process_message("Create a rule for web_url checking for !")

# Get results
print(result["message"])      # Assistant response
print(result["rules"])        # Generated CSV rules
print(result["json_rules"])   # Generated JSON rules
print(result["code"])         # Generated Python code
print(result["clarification"]) # Clarification request

# Get all rules as CSV
csv_content = manager.get_rules_csv()

# Reset conversation
manager.reset()
```

### FieldMatcher

```python
from app.field_matcher import FieldMatcher

matcher = FieldMatcher()

# Find matching fields
results = matcher.match("web_url", limit=5)
# Returns: [(field_name, score, label), ...]

# Match multiple
matches = matcher.match_multiple(["email", "url", "address"])

# List tables
tables = matcher.list_tables()

# Get field info
info = matcher.get_field_info("acnc.bvd_id_number")
```

### RuleGenerator

```python
from app.rule_generator import RuleGenerator

generator = RuleGenerator()

# Create rules
rule = generator.create_regex_rule(
    field_name="web_url",
    pattern="^[^!]*$",
    error_message="URL contains !",
    severity="ERROR"
)

# Save to file
filepath = generator.save_rules([rule])
```

## Support

- Issues: https://github.com/joelk111/java-jersey-webservices/issues
- Documentation: See `/docs` folder
