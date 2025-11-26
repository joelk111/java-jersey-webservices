#!/bin/bash
# NLP Rules Engine - Test Script
# Runs unit tests and generates coverage report

set -e

echo "=========================================="
echo "NLP Rules Engine - Test Suite"
echo "=========================================="

# Activate virtual environment if exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Run tests with coverage
echo ""
echo "Running unit tests..."
python -m pytest tests/ -v --cov=app --cov-report=html --cov-report=term-missing

echo ""
echo "=========================================="
echo "Tests Complete!"
echo "=========================================="
echo ""
echo "Coverage report generated in htmlcov/index.html"
