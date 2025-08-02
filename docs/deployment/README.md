# Deployment Guide

This document provides comprehensive deployment instructions for the Self-Evolving MoE-Router.

## Quick Start

### Docker Deployment

```bash
# Build the production image
make docker-build

# Run with docker-compose (recommended)
docker-compose up -d

# Or run standalone container
docker run -d --name moe-router \
  --gpus all \
  -p 8080:8080 \
  -v ./checkpoints:/app/checkpoints \
  -v ./data:/app/data \
  self-evolving-moe-router:latest
```

### Development Environment

```bash
# Start development environment with all services
make docker-compose-full

# Access services:
# - Application: http://localhost:8080
# - Jupyter: http://localhost:8888
# - TensorBoard: http://localhost:6006
# - MLflow: http://localhost:5000
# - Grafana: http://localhost:3000
```

## Build Configurations

### Production Build

The production Docker image is optimized for:
- Minimal attack surface (distroless base)
- Non-root execution
- Multi-stage build for reduced size
- Security scanning with health checks

```dockerfile
# Production-ready image
FROM python:3.11-slim-bullseye as production
# ... (see Dockerfile for full configuration)
```

### Development Build

Development image includes:
- Full development dependencies
- Jupyter Lab
- Debugging tools
- Source code mounting

### Security Features

- Non-root user execution
- Minimal base image
- Security vulnerability scanning
- Secrets management via environment variables
- Read-only filesystem where possible

## Environment Variables

### Core Configuration

```bash
# Python environment
PYTHONPATH=/app/src
PYTHONUNBUFFERED=1
PYTHONDONTWRITEBYTECODE=1

# CUDA/GPU configuration
CUDA_VISIBLE_DEVICES=0

# Logging and monitoring
WANDB_MODE=offline
MLFLOW_TRACKING_URI=http://mlflow:5000

# Security
JUPYTER_TOKEN=your-secure-token-here
```

### Application-specific

```bash
# Evolution parameters
EVOLUTION_POPULATION_SIZE=100
EVOLUTION_GENERATIONS=1000
MUTATION_RATE=0.1

# Model configuration
MAX_EXPERTS=32
DEFAULT_SPARSITY=0.1

# Resource limits
MAX_MEMORY_GB=16
TARGET_LATENCY_MS=10
```

## Health Checks

### Application Health

The application includes built-in health checks:

```python
# Health check endpoint
GET /health
{
  "status": "healthy",
  "version": "0.1.0",
  "evolution_status": "running",
  "active_experts": 16,
  "current_generation": 42
}
```

### Docker Health Check

```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import self_evolving_moe; print('OK')" || exit 1
```

## Scaling and Performance

### Horizontal Scaling

```yaml
# docker-compose scaling
version: '3.8'
services:
  moe-router:
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 16G
          cpus: '4'
        reservations:
          memory: 8G
          cpus: '2'
```

### GPU Support

```yaml
services:
  moe-router:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### Memory Optimization

- Use memory-mapped model loading
- Configure gradient checkpointing
- Enable mixed precision training
- Implement dynamic batching

## Monitoring and Observability

### Metrics Collection

Prometheus metrics are exposed on `/metrics`:

```
# Evolution metrics
evolution_generation_total
evolution_fitness_best
evolution_diversity_score

# Performance metrics
inference_latency_seconds
memory_usage_bytes
gpu_utilization_percent

# Expert metrics
expert_utilization_ratio
routing_efficiency_score
load_balance_variance
```

### Logging Configuration

```python
# Structured logging
import structlog

logger = structlog.get_logger("self_evolving_moe")
logger.info("Evolution step completed", 
           generation=42, 
           best_fitness=0.95,
           population_diversity=0.7)
```

### Distributed Tracing

```python
# OpenTelemetry integration
from opentelemetry import trace

tracer = trace.get_tracer(__name__)

with tracer.start_as_current_span("evolution_step"):
    # Evolution logic
    pass
```

## Production Considerations

### Security Checklist

- [ ] Run as non-root user
- [ ] Enable security scanning in CI/CD
- [ ] Use secrets management for sensitive data
- [ ] Enable network segmentation
- [ ] Implement authentication and authorization
- [ ] Regular security updates

### Performance Optimization

- [ ] Enable GPU acceleration
- [ ] Configure optimal batch sizes
- [ ] Implement model pruning
- [ ] Use quantization where appropriate
- [ ] Enable gradient compression
- [ ] Optimize data loading pipelines

### Reliability

- [ ] Implement circuit breakers
- [ ] Add retry mechanisms
- [ ] Configure graceful shutdowns
- [ ] Implement backup and recovery
- [ ] Set up monitoring and alerting
- [ ] Document runbooks

## Troubleshooting

### Common Issues

#### Out of Memory

```bash
# Check memory usage
docker stats moe-router-app

# Reduce batch size or model size
export MAX_BATCH_SIZE=16
export NUM_EXPERTS=16
```

#### GPU Not Available

```bash
# Check GPU availability
docker run --rm --gpus all nvidia/cuda:11.8-devel-ubuntu20.04 nvidia-smi

# Verify CUDA installation
python -c "import torch; print(torch.cuda.is_available())"
```

#### Slow Evolution

```bash
# Check evolution progress
curl http://localhost:8080/metrics | grep evolution_generation

# Tune evolution parameters
export POPULATION_SIZE=50  # Reduce for faster evolution
export MUTATION_RATE=0.2   # Increase for more exploration
```

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# Run with debugging
docker-compose -f docker-compose.yml -f docker-compose.debug.yml up
```

### Performance Profiling

```bash
# Profile application
make profile

# Memory profiling
python -m memory_profiler examples/basic_evolution.py

# GPU profiling
nvprof python examples/basic_evolution.py
```

## Deployment Environments

### Development

```bash
# Quick development setup
make docker-compose-dev
```

### Staging

```bash
# Staging environment with monitoring
docker-compose --profile dev --profile monitoring up -d
```

### Production

```bash
# Production deployment
docker-compose --profile production up -d

# With full observability stack
docker-compose --profile production --profile monitoring --profile database up -d
```

## Backup and Recovery

### Model Checkpoints

```bash
# Backup evolved models
tar -czf checkpoints-$(date +%Y%m%d).tar.gz checkpoints/

# Restore from backup
tar -xzf checkpoints-20241201.tar.gz
```

### Database Backup

```bash
# PostgreSQL backup
docker exec moe-postgres pg_dump -U moe_user moe_experiments > backup.sql

# Restore database
docker exec -i moe-postgres psql -U moe_user moe_experiments < backup.sql
```

### Volume Management

```bash
# List volumes
docker volume ls | grep moe

# Backup volume
docker run --rm -v moe_postgres_data:/data -v $(pwd):/backup alpine \
  tar czf /backup/postgres_backup.tar.gz -C /data .

# Restore volume
docker run --rm -v moe_postgres_data:/data -v $(pwd):/backup alpine \
  tar xzf /backup/postgres_backup.tar.gz -C /data
```

## Migration Guide

### Version Upgrades

1. Stop services: `docker-compose down`
2. Backup data: `make backup`
3. Pull new image: `docker-compose pull`
4. Run migrations: `make migrate`
5. Start services: `docker-compose up -d`

### Schema Migrations

```python
# Evolution schema migration
from self_evolving_moe.migrations import migrate_topology_schema

migrate_topology_schema(
    from_version="0.1.0",
    to_version="0.2.0",
    checkpoint_path="checkpoints/"
)
```

## Support and Resources

### Documentation Links

- [Architecture Overview](../ARCHITECTURE.md)
- [Development Guide](../DEVELOPMENT.md)
- [API Reference](../api/)
- [Troubleshooting Guide](../troubleshooting/)

### Community Support

- GitHub Issues: [Report bugs and request features](../../issues)
- Discussions: [Community discussions](../../discussions)
- Discord: [Join our Discord server](https://discord.gg/moe-router)

### Professional Support

For enterprise deployments and professional support:
- Email: support@terragon-labs.com
- Documentation: [Enterprise Guide](../enterprise/)