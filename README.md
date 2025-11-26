# NLP Rules Engine

A natural language processing system for creating data quality validation rules. Describe your validation needs in plain English, and the system generates rules in CSV or JSON format.

## Features

- **Natural Language Input**: Describe rules conversationally
- **Fuzzy Field Matching**: Matches 5,095+ ORBIS data dictionary fields
- **Automatic Clarification**: Asks follow-up questions when info is missing
- **Multiple Output Formats**: CSV rules and JSON for complex validations
- **Custom Function Generation**: Creates Python code for complex validations
- **Local LLM**: Runs on Llama 3.1 8B via Ollama (no cloud API required)

## Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/joelk111/java-jersey-webservices.git
cd java-jersey-webservices
./scripts/setup.sh

# 2. Start Ollama and pull model
ollama serve &
ollama pull llama3.1:8b-instruct-q4_0

# 3. Run
source venv/bin/activate
python -m app.main
```

Open http://localhost:7860 in your browser.

## Docker Deployment

```bash
# Production
docker-compose up -d

# Development (with hot-reload)
docker-compose -f docker-compose.dev.yml up
```

## Example Usage

```
User: Create a rule for web_url that checks if it contains exclamation points and fail the test

System: Created rule:
rule_id,rule_name,rule_type,field_name,validation_type,validation_value,error_message,severity,is_active
R20251126_0001,Regex_Check_web_url,REGEX,all_addresses.web_url,REGEX,"^[^!]*$",URL contains exclamation point,ERROR,TRUE
```

```
User: Create a rule that checks if a field is NULL

System: I can create a NOT_NULL rule, but which field should be required?
```

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 16 GB | 32 GB |
| GPU VRAM | 6 GB | 8+ GB |
| Python | 3.10 | 3.11+ |
| Storage | 10 GB | 20 GB |

## Documentation

- [System Overview](docs/01_SYSTEM_OVERVIEW.md)
- [Installation Guide](docs/02_INSTALLATION_SETUP.md)
- [Development Guide](docs/DEVELOPER_GUIDE.md)
- [Testing & QA](docs/04_TESTING_QA.md)
- [Rules FAQ](RULES_FAQ.md)

## Rule Types Supported

| Type | Description | Example |
|------|-------------|---------|
| REGEX | Pattern matching | Check for special characters |
| NOT_NULL | Required field | Ensure field has value |
| LENGTH | String length | Must be 2 characters |
| RANGE | Numeric range | Between 0 and 100 |
| IN_LIST | Allowed values | Active, Inactive, Pending |
| CUSTOM_FUNCTION | Python function | Complex validation |

## Project Structure

```
├── app/                    # Application code
├── config/                 # Environment configs
├── tests/                  # Test suite
├── scripts/                # Setup scripts
├── docs/                   # Documentation
├── Dockerfile              # Production image
├── docker-compose.yml      # Docker deployment
├── orbis_field_names.c     # Field dictionary
└── requirements.txt        # Dependencies
```

## License

MIT
