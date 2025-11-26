# NLP Rules Engine - Dockerfile
# Multi-stage build for optimized image size

# Build stage
FROM python:3.11-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local

# Ensure scripts are in PATH
ENV PATH=/root/.local/bin:$PATH

# Copy application code
COPY app/ ./app/
COPY config/ ./config/
COPY orbis_field_names.c ./data/
COPY sample_rules.csv ./data/
COPY sample_rules_json.csv ./data/
COPY rules.py ./data/
COPY RULES_FAQ.md ./docs/

# Create output directories
RUN mkdir -p /app/output/rules /app/output/functions /app/logs

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV ENVIRONMENT=production
ENV FIELD_DICTIONARY_PATH=/app/data/orbis_field_names.c
ENV RULES_OUTPUT_DIR=/app/output/rules
ENV CUSTOM_FUNCTIONS_DIR=/app/output/functions

# Expose port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# Run application
CMD ["python", "-m", "app.main"]
