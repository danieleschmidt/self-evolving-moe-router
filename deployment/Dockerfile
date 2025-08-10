# TERRAGON Production Deployment - Multi-stage Docker Build
# Self-Evolving MoE-Router Production Container

FROM python:3.11-slim as builder

# Build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create application directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM python:3.11-slim

# Production dependencies
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -r terragon && useradd -r -g terragon terragon

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local

# Create application structure
WORKDIR /app
RUN mkdir -p /app/logs /app/data /app/models /app/config

# Copy application code
COPY src/ ./src/
COPY *.py ./
COPY config/ ./config/
COPY deployment/entrypoint.sh ./entrypoint.sh

# Set permissions
RUN chown -R terragon:terragon /app && chmod +x entrypoint.sh

# Health check script
COPY deployment/healthcheck.py ./healthcheck.py
RUN chmod +x healthcheck.py

# Environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV TERRAGON_ENV=production
ENV TERRAGON_LOG_LEVEL=INFO
ENV TERRAGON_WORKERS=4
ENV TERRAGON_PORT=8080

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python healthcheck.py

# Switch to non-root user
USER terragon

# Entry point
ENTRYPOINT ["./entrypoint.sh"]
CMD ["production"]