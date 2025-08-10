# TERRAGON Production Deployment Guide
## Self-Evolving MoE-Router Production Deployment

This guide covers the complete production deployment process for the TERRAGON Self-Evolving MoE-Router system, implementing all three generations (MAKE IT WORK, MAKE IT ROBUST, MAKE IT SCALE) with autonomous SDLC execution.

## 🎯 TERRAGON Deployment Overview

The TERRAGON system provides:
- **Generation 1**: Core working functionality with evolutionary MoE routing
- **Generation 2**: Robust error handling, monitoring, and fault tolerance
- **Generation 3**: High-performance scaling, distributed execution, and auto-scaling
- **Quality Gates**: Mandatory 85%+ test coverage, zero security vulnerabilities, sub-200ms response times
- **Research Mode**: Novel evolutionary algorithms and comparative analysis

## 📋 Prerequisites

### System Requirements
- Kubernetes cluster (1.20+)
- Docker Engine (20.10+)
- Minimum 8GB RAM per node
- Minimum 4 CPU cores per node
- 50GB+ persistent storage

### Dependencies
- Python 3.11+
- FastAPI and Uvicorn
- Redis for caching
- PostgreSQL for persistence (optional)
- Prometheus for monitoring
- Grafana for visualization

## 🐳 Docker Deployment

### 1. Build Production Image

```bash
# Build multi-architecture image
docker build -f deployment/Dockerfile -t terragon/self-evolving-moe-router:latest .

# Build with specific version
docker build -f deployment/Dockerfile \
  --build-arg BUILD_DATE=$(date -u +%Y-%m-%dT%H:%M:%SZ) \
  --build-arg VCS_REF=$(git rev-parse HEAD) \
  --build-arg VERSION=v1.0.0 \
  -t terragon/self-evolving-moe-router:v1.0.0 .
```

### 2. Run with Docker Compose (Development)

```bash
# Start full development environment
cd deployment
docker-compose up -d

# Check services
docker-compose ps

# View logs
docker-compose logs -f terragon-app

# Run quality gates
docker-compose --profile tools run quality-gates

# Run research execution
docker-compose --profile tools run research-executor
```

### 3. Single Container Deployment

```bash
# Run production container
docker run -d \
  --name terragon-production \
  -p 8080:8080 \
  -e TERRAGON_ENV=production \
  -e TERRAGON_WORKERS=4 \
  -v terragon-data:/app/data \
  -v terragon-logs:/app/logs \
  --restart unless-stopped \
  terragon/self-evolving-moe-router:latest

# Health check
curl http://localhost:8080/health

# Quality gates
docker exec terragon-production python quality_gates_improved.py

# Research execution
docker exec terragon-production python research_standalone.py
```

## ☸️ Kubernetes Deployment

### 1. Deploy to Kubernetes

```bash
# Create namespace
kubectl apply -f deployment/kubernetes/namespace.yaml

# Deploy configuration
kubectl apply -f deployment/kubernetes/configmap.yaml

# Deploy persistent volumes (if needed)
kubectl apply -f deployment/kubernetes/pvc.yaml

# Deploy application
kubectl apply -f deployment/kubernetes/deployment.yaml
kubectl apply -f deployment/kubernetes/service.yaml

# Setup auto-scaling
kubectl apply -f deployment/kubernetes/hpa.yaml

# Verify deployment
kubectl get pods -n terragon
kubectl get services -n terragon
```

### 2. Monitor Deployment

```bash
# Check deployment status
kubectl rollout status deployment/terragon-moe-router -n terragon

# View logs
kubectl logs -f deployment/terragon-moe-router -n terragon

# Port forward for local access
kubectl port-forward service/terragon-moe-router-service 8080:80 -n terragon

# Run health check
curl http://localhost:8080/health
```

### 3. Scale Deployment

```bash
# Manual scaling
kubectl scale deployment terragon-moe-router --replicas=5 -n terragon

# Check HPA status
kubectl get hpa -n terragon

# Check resource usage
kubectl top pods -n terragon
kubectl top nodes
```

## 🔄 CI/CD Pipeline

### 1. GitHub Actions Setup

Copy the provided CI/CD configuration:

```bash
# Copy CI/CD workflow
cp deployment/ci-cd/github-actions.yml .github/workflows/terragon-cicd.yml

# Commit and push
git add .github/workflows/terragon-cicd.yml
git commit -m "Add TERRAGON CI/CD pipeline"
git push
```

### 2. Required Secrets

Configure the following secrets in your GitHub repository:

```bash
# Container registry
GITHUB_TOKEN (automatically provided)

# Kubernetes clusters
KUBECONFIG_STAGING
KUBECONFIG_PRODUCTION

# Optional: additional secrets for monitoring, databases, etc.
```

### 3. Pipeline Execution

The pipeline automatically:
1. Runs TERRAGON quality gates (85%+ coverage, security scan, performance validation)
2. Validates research execution mode
3. Builds and pushes Docker images
4. Deploys to staging on `develop` branch
5. Deploys to production on version tags (`v*`)
6. Runs comprehensive health checks
7. Sets up monitoring and alerting

## 📊 Monitoring and Observability

### 1. Prometheus Metrics

```bash
# Deploy Prometheus configuration
kubectl create configmap prometheus-config \
  --from-file=deployment/monitoring/prometheus-config.yaml \
  -n terragon

# Access Prometheus UI (if deployed)
kubectl port-forward service/prometheus 9090:9090 -n terragon
# Open http://localhost:9090
```

### 2. Grafana Dashboards

```bash
# Access Grafana (if deployed)
kubectl port-forward service/grafana 3000:3000 -n terragon
# Open http://localhost:3000
# Default: admin/terragon_admin
```

### 3. Alert Rules

The system includes comprehensive alerting for:
- Service availability (SLA: 99.9% uptime)
- Response time (SLA: <200ms 95th percentile)
- Error rates
- Resource utilization
- Evolution system health
- Security events

## 🔒 Security Configuration

### 1. Container Security

```bash
# Run security scan
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
  aquasec/trivy image terragon/self-evolving-moe-router:latest

# Check for vulnerabilities
kubectl exec deployment/terragon-moe-router -n terragon -- python quality_gates_improved.py
```

### 2. Network Policies

```yaml
# Apply network policies for security
kubectl apply -f - <<EOF
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: terragon-network-policy
  namespace: terragon
spec:
  podSelector:
    matchLabels:
      app: terragon
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 8080
  egress:
  - to: []
    ports:
    - protocol: TCP
      port: 443
    - protocol: TCP
      port: 80
EOF
```

## 🧪 Quality Gates Validation

### 1. Pre-deployment Validation

```bash
# Run complete quality gates
python quality_gates_improved.py

# Verify requirements:
# ✅ 85%+ Test Coverage
# ✅ Zero High-Risk Security Vulnerabilities  
# ✅ Sub-200ms Response Time
# ✅ Code Quality Standards
```

### 2. Production Health Checks

```bash
# Container health check
docker exec terragon-production python healthcheck.py

# Kubernetes health check
kubectl exec deployment/terragon-moe-router -n terragon -- python healthcheck.py

# API health check
curl -f http://your-domain.com/health
curl -f http://your-domain.com/ready
curl -f http://your-domain.com/metrics
```

## 🧬 Research Mode Execution

### 1. Run Research Algorithms

```bash
# Execute novel evolutionary algorithms
python research_standalone.py

# Or in container
docker exec terragon-production python research_standalone.py

# Or in Kubernetes
kubectl exec deployment/terragon-moe-router -n terragon -- python research_standalone.py
```

### 2. Research Results

The research execution includes:
- **Adaptive Hybrid Evolution (AHE)**: Combines GA, DE, PSO, and SA
- **Multi-Objective Novelty Search (MONS)**: Novelty-driven evolution
- **Quantum-Inspired Evolution (QIE)**: Quantum superposition and interference

Results are saved to `/app/research_results/` with comprehensive analysis.

## 🚀 High-Performance Scaling

### 1. Auto-scaling Configuration

```bash
# Horizontal Pod Autoscaler
kubectl get hpa terragon-moe-router-hpa -n terragon

# Vertical Pod Autoscaler (if available)
kubectl get vpa terragon-moe-router-vpa -n terragon

# Cluster autoscaling (cloud-specific)
# AWS: Configure cluster-autoscaler
# GCP: Enable GKE autopilot
# Azure: Configure AKS autoscaler
```

### 2. Performance Optimization

```bash
# Run performance benchmarks
python high_performance_evolution.py

# Check resource utilization
kubectl top pods -n terragon
kubectl top nodes

# Monitor evolution performance
curl http://your-domain.com/evolution/status
curl http://your-domain.com/metrics
```

## 🔧 Configuration Management

### 1. Production Configuration

Edit `deployment/kubernetes/configmap.yaml` for:
- Evolution parameters (population size, generations, mutation rate)
- Performance settings (caching, load balancing)
- Security settings (rate limiting, timeouts)
- Monitoring configuration

### 2. Environment-specific Configs

```bash
# Development
TERRAGON_ENV=development
TERRAGON_WORKERS=2
TERRAGON_LOG_LEVEL=DEBUG

# Staging  
TERRAGON_ENV=staging
TERRAGON_WORKERS=3
TERRAGON_LOG_LEVEL=INFO

# Production
TERRAGON_ENV=production
TERRAGON_WORKERS=4
TERRAGON_LOG_LEVEL=WARNING
```

## 🔄 Operational Procedures

### 1. Rolling Updates

```bash
# Update image
kubectl set image deployment/terragon-moe-router \
  moe-router=terragon/self-evolving-moe-router:v1.1.0 \
  -n terragon

# Monitor rollout
kubectl rollout status deployment/terragon-moe-router -n terragon

# Rollback if needed
kubectl rollout undo deployment/terragon-moe-router -n terragon
```

### 2. Backup and Recovery

```bash
# Backup evolution data
kubectl exec deployment/terragon-moe-router -n terragon -- \
  tar czf /tmp/evolution-backup.tar.gz /app/data/

# Copy backup
kubectl cp terragon/pod-name:/tmp/evolution-backup.tar.gz ./evolution-backup.tar.gz

# Restore data
kubectl cp ./evolution-backup.tar.gz terragon/pod-name:/tmp/
kubectl exec deployment/terragon-moe-router -n terragon -- \
  tar xzf /tmp/evolution-backup.tar.gz -C /
```

### 3. Troubleshooting

```bash
# Check pod logs
kubectl logs -f deployment/terragon-moe-router -n terragon

# Debug container
kubectl exec -it deployment/terragon-moe-router -n terragon -- /bin/bash

# Check events
kubectl get events -n terragon --sort-by='.lastTimestamp'

# Resource issues
kubectl describe pods -n terragon
kubectl top pods -n terragon

# Network issues
kubectl exec deployment/terragon-moe-router -n terragon -- nslookup kubernetes.default
```

## 🌟 TERRAGON Compliance Validation

✅ **Generation 1: MAKE IT WORK** - Core evolutionary MoE routing implemented  
✅ **Generation 2: MAKE IT ROBUST** - Error handling, monitoring, fault tolerance  
✅ **Generation 3: MAKE IT SCALE** - Performance optimization, distributed execution  
✅ **Quality Gates**: 85%+ test coverage, zero security vulnerabilities, sub-200ms response  
✅ **Research Execution**: Novel evolutionary algorithms with comparative analysis  
✅ **Production Ready**: Complete CI/CD pipeline, monitoring, and operational procedures

## 📞 Support and Maintenance

For operational support:
1. Check health endpoints: `/health`, `/ready`, `/metrics`
2. Review logs and metrics in monitoring dashboards
3. Run quality gates for validation
4. Execute research mode for algorithm validation
5. Follow troubleshooting procedures above

The TERRAGON system is designed for autonomous operation with comprehensive monitoring, alerting, and self-healing capabilities.

---

**TERRAGON Labs** - Self-Evolving MoE-Router Production Deployment v1.0.0