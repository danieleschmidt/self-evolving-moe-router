# Multi-stage Dockerfile for self-evolving-moe-router
# Optimized for production with security best practices

# Build stage
FROM python:3.11-slim-bullseye as builder

# Security: Create non-root user early
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements files
COPY pyproject.toml README.md ./
COPY src/ src/

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -e ".[dev]" && \
    pip install --no-cache-dir -e ".[viz,distributed,benchmark]"

# Production stage
FROM python:3.11-slim-bullseye as production

# Security: Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set working directory
WORKDIR /app

# Copy Python environment from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY --chown=appuser:appuser src/ src/
COPY --chown=appuser:appuser configs/ configs/
COPY --chown=appuser:appuser examples/ examples/
COPY --chown=appuser:appuser pyproject.toml README.md ./

# Create necessary directories
RUN mkdir -p /app/checkpoints /app/logs /app/data && \
    chown -R appuser:appuser /app

# Security: Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import self_evolving_moe; print('OK')" || exit 1

# Expose default port for visualization dashboard
EXPOSE 8080

# Set default environment variables
ENV PYTHONPATH="/app/src:$PYTHONPATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Default command
CMD ["python", "-m", "self_evolving_moe.cli", "--help"]