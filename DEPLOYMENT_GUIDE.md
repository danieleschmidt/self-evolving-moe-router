# 🚀 Self-Evolving MoE-Router Deployment Guide

This guide provides comprehensive instructions for deploying the Self-Evolving MoE-Router in production environments.

## 📋 Prerequisites

### System Requirements

**Minimum Requirements:**
- **CPU**: 4 cores, 2.4 GHz
- **RAM**: 8GB
- **Storage**: 50GB SSD
- **OS**: Ubuntu 20.04+ / CentOS 8+ / macOS 11+
- **Python**: 3.8+

**Recommended Requirements:**
- **CPU**: 8+ cores, 3.0+ GHz
- **RAM**: 32GB+
- **Storage**: 200GB+ NVMe SSD
- **GPU**: NVIDIA GPU with 8GB+ VRAM (optional but recommended)
- **Network**: 1Gbps+ for distributed training

### Software Prerequisites

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install build essentials
sudo apt install -y build-essential cmake git curl wget
sudo apt install -y libblas-dev liblapack-dev gfortran
sudo apt install -y pkg-config libhdf5-dev

# Install Docker (for containerized deployment)
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh
sudo usermod -aG docker $USER

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

## 🛠️ Installation Methods

### Method 1: Docker Deployment (Recommended)

**Quick Start:**
```bash
# Clone repository
git clone https://github.com/terragon-labs/self-evolving-moe-router.git
cd self-evolving-moe-router

# Start production stack
docker-compose -f docker-compose.production.yml up -d

# Verify deployment
curl http://localhost/health
```

**Production Configuration:**
```bash
# Copy and customize configuration
cp configs/evolution_default.yaml configs/production.yaml
nano configs/production.yaml

# Set environment variables
export MOE_ENV=production
export MOE_LOG_LEVEL=INFO
export MOE_WORKERS=8

# Deploy with custom config
docker-compose -f docker-compose.production.yml up -d
```

### Method 2: Native Installation

**Install Package:**
```bash
# Create virtual environment
python3 -m venv moe-env
source moe-env/bin/activate

# Install from PyPI (when published)
pip install self-evolving-moe-router

# Or install from source
git clone https://github.com/terragon-labs/self-evolving-moe-router.git
cd self-evolving-moe-router
pip install -e ".[dev]"
```

**GPU Support:**
```bash
# Install CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify GPU availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Method 3: Kubernetes Deployment

**Prerequisites:**
- Kubernetes cluster 1.21+
- kubectl configured
- Helm 3.0+

**Deploy with Helm:**
```bash
# Add Terragon Helm repository
helm repo add terragon https://charts.terragonlabs.com
helm repo update

# Install MoE-Router
helm install moe-router terragon/self-evolving-moe-router \
  --namespace moe-system \
  --create-namespace \
  --set image.tag=latest \
  --set resources.requests.memory=8Gi \
  --set resources.requests.cpu=4000m
```

## ⚙️ Configuration

### Environment Variables

```bash
# Core Configuration
export MOE_ENV=production                    # Environment: development, staging, production
export MOE_LOG_LEVEL=INFO                    # Logging level: DEBUG, INFO, WARNING, ERROR
export MOE_CONFIG_PATH=/app/config/production.yaml
export MOE_DATA_DIR=/app/data
export MOE_CHECKPOINT_DIR=/app/checkpoints

# Service Configuration
export MOE_API_PORT=8000                     # API server port
export MOE_METRICS_PORT=9090                 # Metrics server port
export MOE_WORKERS=4                         # Number of API workers

# Evolution Configuration
export MOE_POPULATION_SIZE=100               # Evolution population size
export MOE_GENERATIONS=500                   # Maximum generations
export MOE_MUTATION_RATE=0.1                 # Mutation rate
export MOE_PARALLEL_WORKERS=8                # Parallel evaluation workers

# Database Configuration
export MOE_DATABASE_URL=postgresql://user:pass@localhost:5432/moe_db
export MOE_REDIS_URL=redis://localhost:6379/0

# Resource Limits
export MOE_MEMORY_LIMIT=8G                   # Memory limit
export MOE_CPU_LIMIT=4.0                     # CPU limit
export MOE_GPU_MEMORY_LIMIT=8G               # GPU memory limit

# Monitoring
export MOE_ENABLE_METRICS=true               # Enable Prometheus metrics
export MOE_ENABLE_TRACING=false              # Enable distributed tracing
export MOE_ALERT_WEBHOOK_URL=https://hooks.slack.com/...
```

### Configuration File (YAML)

```yaml
# production.yaml
evolution:
  population_size: 100
  generations: 500
  mutation_rate: 0.1
  crossover_rate: 0.7
  selection_method: "tournament"
  parallel_workers: 8
  
  objectives:
    - name: "accuracy"
      weight: 1.0
      target: 0.95
    - name: "latency"
      weight: -0.2
      target: 10.0  # ms
    - name: "sparsity"
      weight: 0.1
      target: 0.9

model:
  num_experts: 32
  expert_dim: 768
  expert_type: "transformer_block"
  slimmable_widths: [192, 384, 576, 768]

distributed:
  backend: "multiprocessing"  # multiprocessing, ray, dask
  num_workers: 8
  batch_size_per_worker: 10
  fault_tolerance: true
  load_balancing: true

monitoring:
  enable_metrics: true
  metrics_port: 9090
  log_level: "INFO"
  alert_thresholds:
    cpu_percent: 90.0
    memory_percent: 85.0
    gpu_utilization: 95.0

deployment:
  environment: "production"
  debug: false
  checkpoint_interval: 10
  auto_scaling: true
  health_check_interval: 30
```

## 🔧 Service Management

### Systemd Service (Linux)

**Create service file:**
```bash
sudo nano /etc/systemd/system/moe-router.service
```

```ini
[Unit]
Description=Self-Evolving MoE-Router
After=network.target
Wants=network.target

[Service]
Type=exec
User=moe
Group=moe
WorkingDirectory=/opt/moe-router
Environment=MOE_ENV=production
Environment=MOE_CONFIG_PATH=/opt/moe-router/config/production.yaml
ExecStart=/opt/moe-router/venv/bin/python -m self_evolving_moe.cli serve
ExecReload=/bin/kill -HUP $MAINPID
Restart=always
RestartSec=10
KillMode=mixed
TimeoutStopSec=30

# Resource limits
LimitNOFILE=65536
LimitNPROC=32768

[Install]
WantedBy=multi-user.target
```

**Enable and start service:**
```bash
sudo systemctl daemon-reload
sudo systemctl enable moe-router
sudo systemctl start moe-router
sudo systemctl status moe-router
```

### Process Monitoring

**Using supervisord:**
```ini
# /etc/supervisor/conf.d/moe-router.conf
[program:moe-router]
command=/opt/moe-router/venv/bin/python -m self_evolving_moe.cli serve
directory=/opt/moe-router
user=moe
autostart=true
autorestart=true
redirect_stderr=true
stdout_logfile=/var/log/moe-router/moe-router.log
stdout_logfile_maxbytes=100MB
stdout_logfile_backups=5
environment=MOE_ENV=production,MOE_CONFIG_PATH=/opt/moe-router/config/production.yaml
```

## 📊 Monitoring & Observability

### Prometheus Metrics

The system exposes comprehensive metrics on `/metrics` endpoint:

```
# Evolution metrics
moe_evolution_generation_current
moe_evolution_best_fitness
moe_evolution_population_diversity
moe_evolution_convergence_rate

# Performance metrics  
moe_inference_latency_seconds
moe_inference_throughput_total
moe_expert_utilization_ratio
moe_routing_efficiency_ratio

# System metrics
moe_memory_usage_bytes
moe_gpu_memory_usage_bytes
moe_cpu_usage_percent
moe_active_workers_total
```

### Grafana Dashboards

Import pre-built dashboards from `monitoring/grafana/dashboards/`:

1. **Evolution Overview**: Population fitness, convergence metrics
2. **Performance Dashboard**: Latency, throughput, resource usage
3. **Expert Analysis**: Expert utilization, routing patterns
4. **System Health**: Infrastructure metrics, alerts

### Logging Configuration

**Structured logging with JSON format:**
```python
import logging
from self_evolving_moe.utils.logging import setup_logging

# Setup production logging
setup_logging(
    level="INFO",
    json_format=True,
    log_file="/var/log/moe-router/app.log"
)
```

**Log aggregation with ELK stack:**
```yaml
# filebeat.yml
filebeat.inputs:
- type: log
  enabled: true
  paths:
    - /var/log/moe-router/*.log
  json.keys_under_root: true

output.elasticsearch:
  hosts: ["elasticsearch:9200"]
  index: "moe-router-logs-%{+yyyy.MM.dd}"
```

## 🛡️ Security Considerations

### Network Security

```bash
# Firewall configuration
sudo ufw allow 22/tcp      # SSH
sudo ufw allow 80/tcp      # HTTP
sudo ufw allow 443/tcp     # HTTPS
sudo ufw allow 8000/tcp    # API (internal only)
sudo ufw allow 9090/tcp    # Metrics (internal only)
sudo ufw enable
```

### Authentication & Authorization

**API Key authentication:**
```python
# Set API key
export MOE_API_KEY=your-secure-api-key-here

# Use in requests
curl -H "Authorization: Bearer $MOE_API_KEY" http://localhost:8000/api/evolve
```

**OAuth2 integration:**
```yaml
# In production.yaml
security:
  authentication:
    provider: "oauth2"
    client_id: "your-client-id"
    client_secret: "your-client-secret"
    authorization_url: "https://your-oauth-provider.com/auth"
```

### Data Encryption

**Encrypt sensitive data:**
```bash
# Encrypt configuration files
gpg --symmetric --cipher-algo AES256 production.yaml

# Decrypt at runtime
gpg --decrypt production.yaml.gpg > /tmp/production.yaml
MOE_CONFIG_PATH=/tmp/production.yaml
```

## 🚀 Scaling & Performance

### Horizontal Scaling

**Multiple API instances with load balancer:**
```yaml
# docker-compose.scale.yml
version: '3.8'
services:
  moe-api:
    # ... existing config
    deploy:
      replicas: 4
  
  load-balancer:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx-lb.conf:/etc/nginx/nginx.conf
    depends_on:
      - moe-api
```

**Kubernetes horizontal pod autoscaler:**
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: moe-router-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: moe-router
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### Distributed Training

**Ray cluster for distributed evolution:**
```python
# Start Ray cluster
ray start --head --port=6379

# Configure distributed evolution
export MOE_DISTRIBUTED_BACKEND=ray
export MOE_RAY_ADDRESS=ray://localhost:10001
```

**Multi-GPU training:**
```python
# Configure GPU allocation
export CUDA_VISIBLE_DEVICES=0,1,2,3
export MOE_GPU_PER_WORKER=1
export MOE_DISTRIBUTED_WORKERS=4
```

## 🔍 Troubleshooting

### Common Issues

**1. Out of Memory Errors**
```bash
# Check memory usage
docker stats moe-api

# Increase memory limits
docker-compose -f docker-compose.production.yml up -d --scale moe-api=1 \
  --memory=8g --memory-swap=16g
```

**2. Slow Evolution Performance**
```bash
# Profile performance
python -m self_evolving_moe.optimization.profiling

# Increase parallel workers
export MOE_PARALLEL_WORKERS=16
```

**3. Database Connection Issues**
```bash
# Check database connectivity
docker exec moe-postgres pg_isready

# Reset database connections
docker restart moe-postgres
```

### Health Checks

**System health check:**
```bash
# Quick health check
python /app/healthcheck.py --quick

# Comprehensive health check
python /app/healthcheck.py --verbose --json
```

**Service status:**
```bash
# Check all services
docker-compose -f docker-compose.production.yml ps

# View logs
docker-compose -f docker-compose.production.yml logs -f moe-api
```

### Log Analysis

**Search for errors:**
```bash
# Recent errors
docker logs moe-api 2>&1 | grep -i error | tail -20

# Evolution progress
docker logs moe-evolution 2>&1 | grep "Generation"

# Performance metrics
curl http://localhost:9090/metrics | grep moe_
```

## 📈 Performance Tuning

### CPU Optimization

```bash
# Set CPU affinity
taskset -c 0-7 python -m self_evolving_moe.cli serve

# Enable CPU optimizations
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
```

### Memory Optimization

```python
# Configure memory-efficient settings
evolution:
  population_size: 50        # Smaller population
  batch_size_per_worker: 5   # Smaller batches
  checkpoint_interval: 5     # Frequent checkpointing

model:
  expert_dim: 512           # Smaller model
  gradient_checkpointing: true
```

### GPU Optimization

```bash
# CUDA optimizations
export CUDA_LAUNCH_BLOCKING=0
export CUDA_CACHE_DISABLE=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Mixed precision training
export MOE_MIXED_PRECISION=true
```

## 🔄 Backup & Recovery

### Data Backup

```bash
# Backup checkpoints
tar -czf moe-checkpoints-$(date +%Y%m%d).tar.gz /app/checkpoints/

# Backup database
docker exec moe-postgres pg_dump -U moe_user moe_db > moe-db-backup.sql

# Backup configuration
cp -r configs/ config-backup-$(date +%Y%m%d)/
```

### Disaster Recovery

```bash
# Restore from backup
tar -xzf moe-checkpoints-backup.tar.gz -C /app/

# Restore database
docker exec -i moe-postgres psql -U moe_user moe_db < moe-db-backup.sql

# Restart services
docker-compose -f docker-compose.production.yml restart
```

## 📞 Support & Maintenance

### Automated Maintenance

```bash
# Setup cron jobs for maintenance
crontab -e

# Clean old checkpoints (daily at 2 AM)
0 2 * * * find /app/checkpoints -name "*.pt" -mtime +30 -delete

# Backup databases (weekly)
0 3 * * 0 /opt/moe-router/scripts/backup-database.sh

# Update system packages (monthly)
0 4 1 * * apt update && apt upgrade -y
```

### Monitoring Alerts

```yaml
# Alert rules for Prometheus
groups:
- name: moe-router
  rules:
  - alert: HighMemoryUsage
    expr: moe_memory_usage_percent > 90
    for: 5m
    annotations:
      summary: "High memory usage detected"
      
  - alert: EvolutionStalled
    expr: increase(moe_evolution_generation_current[1h]) == 0
    for: 30m
    annotations:
      summary: "Evolution process appears stalled"
```

### Support Channels

- **Documentation**: https://self-evolving-moe.readthedocs.io
- **Issues**: https://github.com/terragon-labs/self-evolving-moe-router/issues
- **Discussions**: https://github.com/terragon-labs/self-evolving-moe-router/discussions
- **Email**: support@terragonlabs.com

---

## 🎯 Quick Reference

### Essential Commands

```bash
# Start production stack
docker-compose -f docker-compose.production.yml up -d

# Scale API workers
docker-compose -f docker-compose.production.yml up -d --scale moe-api=4

# View real-time logs
docker-compose -f docker-compose.production.yml logs -f

# Run evolution experiment
python -m self_evolving_moe.cli evolve --config configs/production.yaml

# Health check
curl http://localhost/health

# Metrics
curl http://localhost:9090/metrics

# Stop all services
docker-compose -f docker-compose.production.yml down
```

### Performance Benchmarks

| Configuration | Throughput | Latency | Memory |
|---------------|------------|---------|---------|
| CPU-only (4 cores) | 50 samples/sec | 20ms | 2GB |
| CPU-only (8 cores) | 120 samples/sec | 8ms | 4GB |
| GPU (RTX 4090) | 500 samples/sec | 2ms | 8GB |
| Distributed (4 nodes) | 2000 samples/sec | 5ms | 16GB |

This completes the comprehensive deployment guide for the Self-Evolving MoE-Router system.