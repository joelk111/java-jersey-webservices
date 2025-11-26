# NLP Rules Engine - Part 2: Installation and Setup Guide

## Prerequisites

### System Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| OS | Ubuntu 20.04+ / macOS 12+ / Windows 10+ | Ubuntu 22.04 |
| Python | 3.10 | 3.11+ |
| RAM | 16 GB | 32 GB |
| Storage | 10 GB free | 20 GB free |
| GPU (optional) | 6 GB VRAM | 8+ GB VRAM |

### Software Prerequisites

- Git
- Python 3.10+
- pip or conda
- curl (for Ollama installation)

## Step 1: Install Ollama

### Linux
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### macOS
```bash
# Using Homebrew
brew install ollama

# Or download from https://ollama.com/download
```

### Windows
Download from: https://ollama.com/download/windows

### Verify Installation
```bash
ollama --version
```

## Step 2: Download Llama Model

### Start Ollama Service
```bash
# Linux/macOS
ollama serve

# Or run as background service
systemctl start ollama  # if installed as service
```

### Pull Llama 3.1 8B Model
```bash
# Full precision (requires more VRAM)
ollama pull llama3.1:8b-instruct-fp16

# Quantized version (recommended for most systems)
ollama pull llama3.1:8b-instruct-q4_0

# For lower-end systems, use 3B model
ollama pull llama3.2:3b-instruct-q4_0
```

### Verify Model Download
```bash
ollama list
```

Expected output:
```
NAME                            SIZE
llama3.1:8b-instruct-q4_0       4.7 GB
```

## Step 3: Clone Repository

```bash
git clone https://github.com/joelk111/java-jersey-webservices.git
cd java-jersey-webservices
```

## Step 4: Create Python Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate (Linux/macOS)
source venv/bin/activate

# Activate (Windows)
.\venv\Scripts\activate
```

## Step 5: Install Python Dependencies

### Create requirements.txt
```bash
cat > requirements.txt << 'EOF'
# Core dependencies
gradio>=4.0.0
httpx>=0.25.0
requests>=2.31.0

# Fuzzy matching
rapidfuzz>=3.5.0

# Data processing
pandas>=2.0.0
openpyxl>=3.1.0

# Configuration
python-dotenv>=1.0.0

# Typing support
typing-extensions>=4.8.0

# Testing
pytest>=7.4.0
pytest-asyncio>=0.21.0

# Code quality
black>=23.0.0
ruff>=0.1.0
mypy>=1.0.0
EOF
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

## Step 6: Configure Environment

### Create .env file
```bash
cat > .env << 'EOF'
# Ollama Configuration
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b-instruct-q4_0

# Application Settings
APP_HOST=0.0.0.0
APP_PORT=7860
DEBUG=false

# Field Dictionary Path
FIELD_DICTIONARY_PATH=./orbis_field_names.c

# Rules Output Directory
RULES_OUTPUT_DIR=./generated_rules

# Logging
LOG_LEVEL=INFO
EOF
```

## Step 7: Generate Field Index

### Option A: From XLSX (if available)
```bash
python3 << 'EOF'
import pandas as pd

df = pd.read_excel('Copy of data_governance_catalog.orbis_meta.field_dictionary.xlsx')
df = df[df['table_name'].notna()]

output = []
output.append('// AUTO-GENERATED - List of all ORBIS field names')
output.append(f'// Total fields: {len(df)}')
output.append(f'// Unique tables: {df["table_name"].nunique()}')
output.append('')
output.append('const char* ORBIS_FIELD_NAMES[] = {')

for table in sorted(df['table_name'].unique()):
    table_df = df[df['table_name'] == table]
    output.append(f'    // Table: {table} ({len(table_df)} fields)')
    for _, row in table_df.iterrows():
        field = row['field_name']
        label = str(row['label']) if pd.notna(row['label']) else ''
        data_type = str(row['data_type']) if pd.notna(row['data_type']) else 'STRING'
        output.append(f'    "{table}.{field}",  // {label} | {data_type}')
    output.append('')

output.append('};')

with open('orbis_field_names.c', 'w') as f:
    f.write('\n'.join(output))

print(f"Generated field index with {len(df)} fields")
EOF
```

### Option B: Verify Existing File
```bash
head -50 orbis_field_names.c
```

## Step 8: Create Directory Structure

```bash
mkdir -p generated_rules
mkdir -p custom_functions
mkdir -p logs
mkdir -p tests
```

## Step 9: Test Ollama Connection

```bash
python3 << 'EOF'
import httpx
import json

# Test Ollama connection
response = httpx.get("http://localhost:11434/api/tags")
if response.status_code == 200:
    models = response.json()
    print("Ollama connected successfully!")
    print("Available models:")
    for model in models.get("models", []):
        print(f"  - {model['name']}")
else:
    print(f"Error: {response.status_code}")

# Test model inference
print("\nTesting model inference...")
response = httpx.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "llama3.1:8b-instruct-q4_0",
        "prompt": "Say 'Hello, Rules Engine!' and nothing else.",
        "stream": False
    },
    timeout=60.0
)
if response.status_code == 200:
    result = response.json()
    print(f"Model response: {result['response']}")
else:
    print(f"Inference error: {response.status_code}")
EOF
```

## Step 10: Run Quick Verification

```bash
python3 << 'EOF'
import os

print("Verifying installation...\n")

checks = [
    ("Python version", "python3 --version"),
    ("Ollama running", "curl -s http://localhost:11434/api/tags"),
    ("Field dictionary", "test -f orbis_field_names.c"),
    ("Rules examples", "test -f sample_rules.csv"),
    ("Custom functions", "test -f rules.py"),
]

all_passed = True
for name, cmd in checks:
    result = os.system(cmd + " > /dev/null 2>&1")
    status = "PASS" if result == 0 else "FAIL"
    print(f"  [{status}] {name}")
    if result != 0:
        all_passed = False

print()
if all_passed:
    print("All checks passed! Ready to run the application.")
else:
    print("Some checks failed. Please review the installation steps.")
EOF
```

## Troubleshooting

### Ollama Not Starting
```bash
# Check if port is in use
lsof -i :11434

# Kill existing process
pkill ollama

# Start fresh
ollama serve
```

### Model Download Issues
```bash
# Check disk space
df -h

# Clear Ollama cache and retry
rm -rf ~/.ollama/models
ollama pull llama3.1:8b-instruct-q4_0
```

### Python Dependency Conflicts
```bash
# Create fresh environment
deactivate
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### GPU Not Detected
```bash
# Check NVIDIA GPU
nvidia-smi

# Check Ollama GPU usage
OLLAMA_DEBUG=1 ollama run llama3.1:8b-instruct-q4_0
```

## Next Steps

Proceed to **Part 3: Development Guide** for:
- Application source code
- API implementation
- UI development
- Custom function creation
