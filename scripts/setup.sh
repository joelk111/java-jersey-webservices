#!/bin/bash
# NLP Rules Engine - Setup Script
# This script sets up the development environment

set -e

echo "=========================================="
echo "NLP Rules Engine - Setup Script"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print status
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

# Check Python version
echo ""
echo "Checking prerequisites..."

if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    print_status "Python found: $PYTHON_VERSION"
else
    print_error "Python 3 not found. Please install Python 3.10+"
    exit 1
fi

# Check if pip is available
if command -v pip3 &> /dev/null; then
    print_status "pip found"
else
    print_error "pip not found. Please install pip"
    exit 1
fi

# Create virtual environment
echo ""
echo "Setting up virtual environment..."

if [ ! -d "venv" ]; then
    python3 -m venv venv
    print_status "Virtual environment created"
else
    print_warning "Virtual environment already exists"
fi

# Activate virtual environment
source venv/bin/activate
print_status "Virtual environment activated"

# Install dependencies
echo ""
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
print_status "Dependencies installed"

# Create output directories
echo ""
echo "Creating directories..."
mkdir -p generated_rules
mkdir -p custom_functions
mkdir -p logs
print_status "Directories created"

# Copy environment config
echo ""
echo "Setting up configuration..."
if [ ! -f ".env" ]; then
    cp config/development.env .env
    print_status "Development configuration copied to .env"
else
    print_warning ".env already exists, skipping"
fi

# Check for Ollama
echo ""
echo "Checking Ollama installation..."
if command -v ollama &> /dev/null; then
    print_status "Ollama found"

    # Check if Ollama is running
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        print_status "Ollama is running"
    else
        print_warning "Ollama is not running. Start with: ollama serve"
    fi
else
    print_warning "Ollama not found. Install from: https://ollama.com/download"
fi

# Print next steps
echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Start Ollama (if not running): ollama serve"
echo "2. Pull the model: ollama pull llama3.1:8b-instruct-q4_0"
echo "3. Activate venv: source venv/bin/activate"
echo "4. Run the app: python -m app.main"
echo ""
echo "Or use Docker:"
echo "  docker-compose up -d"
echo ""
