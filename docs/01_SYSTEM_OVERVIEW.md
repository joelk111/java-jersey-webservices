# NLP Rules Engine - Part 1: System Overview and Requirements

## Executive Summary

This document describes a **Natural Language Processing (NLP) Rules Creation System** that enables users to create data quality rules using conversational language. The system uses a locally-deployed Large Language Model (LLM) to interpret user requests, match field names using fuzzy matching, generate validation rules, and create custom Python functions when needed.

## System Purpose

The NLP Rules Engine allows data quality analysts to:
1. Describe validation rules in plain English
2. Automatically match field names from the ORBIS data dictionary
3. Generate rules in CSV format for the rules engine
4. Create regex patterns and custom Python validation functions
5. Receive clarifying questions when input is ambiguous

## Llama Model Analysis and Recommendation

### Capability Assessment

Based on research, **Llama 3.1 8B** or **Llama 3.2 3B** are suitable for this task:

| Capability | Llama 3.1 8B | Llama 3.2 3B | Required |
|------------|--------------|--------------|----------|
| Function/Tool Calling | Yes (native) | Yes (native) | Yes |
| Multi-turn Conversation | Yes (128K context) | Yes (128K context) | Yes |
| Code Generation | Strong | Good | Yes |
| Regex Generation | Good | Moderate | Yes |
| Clarifying Questions | Yes | Yes | Yes |
| Local Deployment | Yes (Ollama) | Yes (Ollama) | Yes |

### Recommended Model

**Primary: Llama 3.1 8B Instruct** via Ollama

- Best balance of capability and resource usage
- Native tool calling support
- Strong code generation for custom functions
- Can ask clarifying questions naturally

**Alternative: Llama 3.2 3B** for lower resource environments

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 16 GB | 32 GB |
| GPU VRAM | 6 GB | 8+ GB |
| Storage | 10 GB | 20 GB |
| CPU | 4 cores | 8+ cores |

**CPU-only mode is supported but slower (~7-12 tokens/sec)**

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Interface                           │
│                    (Gradio Web Interface)                        │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Conversation Manager                          │
│         (Multi-turn dialogue, context management)                │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      LLM Interface                               │
│              (Ollama + Llama 3.1 8B Instruct)                   │
│                                                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │ Tool: Field │  │ Tool: Rule  │  │ Tool: Code Generator    │  │
│  │   Matcher   │  │  Generator  │  │ (Custom Functions)      │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Field Name Index                              │
│         (Fuzzy matching with rapidfuzz library)                  │
│         Source: orbis_field_names.c (5,095 fields)              │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Output Generator                            │
│   ┌───────────────┐  ┌───────────────┐  ┌───────────────────┐   │
│   │  CSV Rules    │  │  JSON Rules   │  │  Python Code      │   │
│   └───────────────┘  └───────────────┘  └───────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Key Features

### 1. Natural Language Input
Users describe rules conversationally:
- "Check if web_url contains exclamation points and fail the test"
- "Validate that email addresses are properly formatted"
- "Ensure country codes are exactly 2 uppercase letters"

### 2. Fuzzy Field Name Matching
The system matches user-mentioned fields to the ORBIS data dictionary:
- "web url" → `web_url`
- "company email" → `email_address`
- "bvd id" → `bvd_id_number`

### 3. Clarifying Questions
When input is ambiguous, the LLM asks follow-up questions:
- "Which table should this rule apply to?"
- "Should this fail as ERROR or WARNING?"
- "Should empty values pass or fail this check?"

### 4. Custom Function Generation
For complex validations, the system generates Python code:
```python
def validate_url_no_exclamation(value: str) -> Tuple[bool, Optional[str]]:
    if '!' in value:
        return False, "URL contains exclamation point"
    return True, None
```

### 5. Rule Output Formats
- **CSV Format**: Standard comma-separated rules
- **JSON Format**: Complex composite rules
- **Python Code**: Custom validation functions

## Data Sources

### 1. Field Dictionary
**File**: `orbis_field_names.c` (auto-generated from xlsx)
- 5,095 fields across 109 tables
- Format: `table_name.field_name`
- Includes labels and data types

### 2. Existing Rules
**Files**: `sample_rules.csv`, `sample_rules_json.csv`
- Templates for rule generation
- Examples of all rule types

### 3. Custom Functions
**File**: `rules.py`
- Registry of validation functions
- Templates for new functions

## Success Criteria

1. **Accuracy**: >90% correct field matching
2. **Usability**: Users can create rules without technical knowledge
3. **Completeness**: All rule types supported
4. **Performance**: <5 second response time for simple rules
5. **Reliability**: Asks clarifying questions when needed

## Technology Stack

| Component | Technology |
|-----------|------------|
| LLM Runtime | Ollama |
| LLM Model | Llama 3.1 8B Instruct |
| Web UI | Gradio |
| Backend | Python 3.10+ |
| Fuzzy Matching | rapidfuzz |
| HTTP Client | httpx or requests |
| Config | python-dotenv |

## Security Considerations

1. **Local Deployment**: All processing on local machine
2. **No External API Calls**: LLM runs via Ollama locally
3. **Input Validation**: Sanitize all user inputs
4. **Code Generation Safety**: Review generated Python code
5. **Access Control**: Optional authentication for web UI

## Next Steps

See the following documents:
- **Part 2**: Installation and Setup Guide
- **Part 3**: Development Guide
- **Part 4**: Testing and QA
