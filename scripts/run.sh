#!/bin/bash
# NLP Rules Engine - Run Script
# Usage: ./scripts/run.sh [environment]
# Environments: development (default), staging, production

set -e

ENVIRONMENT=${1:-development}
echo "Starting NLP Rules Engine in $ENVIRONMENT mode..."

# Load environment config
if [ -f "config/${ENVIRONMENT}.env" ]; then
    export $(cat config/${ENVIRONMENT}.env | grep -v '^#' | xargs)
    echo "Loaded configuration from config/${ENVIRONMENT}.env"
fi

# Activate virtual environment if exists
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "Virtual environment activated"
fi

# Check Ollama
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "Warning: Ollama is not running at $OLLAMA_HOST"
    echo "Start Ollama with: ollama serve"
    exit 1
fi

# Run application
echo "Starting application on port $APP_PORT..."
python -m app.main
