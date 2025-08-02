# Monitoring & Observability Guide

This document outlines the comprehensive monitoring and observability setup for the Self-Evolving MoE-Router.

## Overview

The monitoring stack includes:
- **Prometheus**: Metrics collection and storage
- **Grafana**: Visualization and dashboards
- **Structured Logging**: Application and system logs
- **Health Checks**: Service availability monitoring
- **Distributed Tracing**: Request flow tracking
- **Performance Metrics**: Evolution and inference monitoring

## Metrics Collection

### Core Evolution Metrics

```python
# Evolution performance metrics
evolution_generation_current = Gauge('evolution_generation_current', 'Current evolution generation')
evolution_fitness_best = Gauge('evolution_fitness_best', 'Best fitness score in current generation')
evolution_fitness_average = Gauge('evolution_fitness_average', 'Average fitness score in current generation')
evolution_diversity_score = Gauge('evolution_diversity_score', 'Population diversity score')
evolution_convergence_rate = Gauge('evolution_convergence_rate', 'Convergence rate over last N generations')

# Population metrics
evolution_population_size = Gauge('evolution_population_size', 'Current population size')
evolution_elite_count = Gauge('evolution_elite_count', 'Number of elite individuals')
evolution_mutation_rate = Gauge('evolution_mutation_rate', 'Current mutation rate')
evolution_crossover_rate = Gauge('evolution_crossover_rate', 'Current crossover rate')
```

### Expert and Routing Metrics

```python
# Expert utilization
expert_utilization_ratio = Gauge('expert_utilization_ratio', 'Expert utilization ratio', ['expert_id'])
expert_load_balance_variance = Gauge('expert_load_balance_variance', 'Load balance variance across experts')
expert_specialization_score = Gauge('expert_specialization_score', 'Expert specialization metric', ['expert_id'])

# Routing efficiency
routing_sparsity_ratio = Gauge('routing_sparsity_ratio', 'Routing matrix sparsity ratio')
routing_efficiency_score = Gauge('routing_efficiency_score', 'Overall routing efficiency')
routing_connections_active = Gauge('routing_connections_active', 'Number of active routing connections')
routing_path_length_avg = Gauge('routing_path_length_avg', 'Average routing path length')
```

### Performance Metrics

```python
# Inference performance
inference_latency_seconds = Histogram('inference_latency_seconds', 'Inference latency in seconds', 
                                    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0])
inference_throughput_samples = Gauge('inference_throughput_samples', 'Inference throughput (samples/sec)')
inference_batch_size = Histogram('inference_batch_size', 'Inference batch size')

# Memory and compute
memory_usage_bytes = Gauge('memory_usage_bytes', 'Memory usage in bytes', ['memory_type'])
gpu_utilization_percent = Gauge('gpu_utilization_percent', 'GPU utilization percentage', ['gpu_id'])
cpu_utilization_percent = Gauge('cpu_utilization_percent', 'CPU utilization percentage')
disk_usage_bytes = Gauge('disk_usage_bytes', 'Disk usage in bytes', ['mount_point'])
```

### Training Metrics

```python
# Training progress
training_loss = Gauge('training_loss', 'Current training loss')
training_accuracy = Gauge('training_accuracy', 'Current training accuracy')
training_learning_rate = Gauge('training_learning_rate', 'Current learning rate')
training_epoch = Gauge('training_epoch', 'Current training epoch')

# Model quality
model_parameter_count = Gauge('model_parameter_count', 'Number of model parameters')
model_flops = Gauge('model_flops', 'Model FLOPs per forward pass')
model_size_bytes = Gauge('model_size_bytes', 'Model size in bytes')
```

## Health Checks

### Application Health Endpoint

```python
from flask import Flask, jsonify
import torch
import psutil
from datetime import datetime

app = Flask(__name__)

@app.route('/health')
def health_check():
    """Comprehensive health check endpoint"""
    try:
        health_status = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "version": "0.1.0",
            "checks": {
                "application": check_application_health(),
                "database": check_database_health(),
                "gpu": check_gpu_health(),
                "memory": check_memory_health(),
                "disk": check_disk_health(),
                "evolution": check_evolution_health()
            }
        }
        
        # Determine overall status
        failed_checks = [name for name, check in health_status["checks"].items() 
                        if not check["healthy"]]
        
        if failed_checks:
            health_status["status"] = "unhealthy"
            health_status["failed_checks"] = failed_checks
            
        return jsonify(health_status), 200 if health_status["status"] == "healthy" else 503
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }), 500

def check_application_health():
    """Check application-specific health"""
    try:
        # Import main modules to verify they load correctly
        import self_evolving_moe
        return {
            "healthy": True,
            "message": "Application modules loaded successfully"
        }
    except Exception as e:
        return {
            "healthy": False,
            "message": f"Application health check failed: {str(e)}"
        }

def check_gpu_health():
    """Check GPU availability and health"""
    try:
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            return {
                "healthy": True,
                "gpu_count": gpu_count,
                "gpu_memory_gb": gpu_memory / (1024**3),
                "cuda_version": torch.version.cuda
            }
        else:
            return {
                "healthy": True,
                "message": "No GPU available (CPU-only mode)"
            }
    except Exception as e:
        return {
            "healthy": False,
            "message": f"GPU health check failed: {str(e)}"
        }

def check_memory_health():
    """Check system memory health"""
    try:
        memory = psutil.virtual_memory()
        return {
            "healthy": memory.percent < 90,  # Healthy if less than 90% used
            "memory_percent": memory.percent,
            "memory_available_gb": memory.available / (1024**3),
            "memory_total_gb": memory.total / (1024**3)
        }
    except Exception as e:
        return {
            "healthy": False,
            "message": f"Memory health check failed: {str(e)}"
        }
```

### Kubernetes Health Checks

```yaml
# kubernetes/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: moe-router
spec:
  template:
    spec:
      containers:
      - name: moe-router
        image: self-evolving-moe-router:latest
        ports:
        - containerPort: 8080
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 60
          periodSeconds: 30
          timeoutSeconds: 10
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "16Gi"
            cpu: "8"
```

## Structured Logging

### Logging Configuration

```python
import structlog
import logging
from datetime import datetime

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger("self_evolving_moe")
```

### Usage Examples

```python
# Evolution logging
logger.info("Evolution generation completed",
           generation=42,
           best_fitness=0.95,
           average_fitness=0.87,
           diversity_score=0.7,
           mutation_rate=0.1,
           population_size=100,
           elapsed_time_seconds=45.3)

# Performance logging
logger.info("Inference completed",
           batch_size=32,
           latency_ms=12.5,
           throughput_samples_per_sec=2560,
           memory_usage_mb=1024,
           gpu_utilization_percent=85,
           active_experts=16)

# Error logging
logger.error("Evolution step failed",
            generation=42,
            error_type="OutOfMemoryError",
            error_message="CUDA out of memory",
            population_size=100,
            batch_size=64,
            gpu_memory_allocated_mb=8192,
            exc_info=True)

# Debug logging
logger.debug("Routing decision made",
            token_id=123,
            expert_ids=[1, 5, 12],
            routing_weights=[0.6, 0.3, 0.1],
            routing_entropy=0.85,
            load_balance_loss=0.02)
```

## Alerting Configuration

### Prometheus Alert Rules

```yaml
# monitoring/alerts/evolution.yml
groups:
- name: evolution.rules
  rules:
  - alert: EvolutionStagnation
    expr: (evolution_fitness_best - evolution_fitness_best offset 10m) < 0.001
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "Evolution showing no improvement"
      description: "Best fitness has not improved in the last 10 minutes"

  - alert: LowPopulationDiversity
    expr: evolution_diversity_score < 0.1
    for: 2m
    labels:
      severity: warning
    annotations:
      summary: "Low population diversity detected"
      description: "Population diversity is {{ $value }}, indicating potential premature convergence"

  - alert: ExpertImbalance
    expr: expert_load_balance_variance > 0.5
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "Expert load imbalance detected"
      description: "Load variance across experts is {{ $value }}, indicating poor load balancing"

  - alert: HighInferenceLatency
    expr: histogram_quantile(0.95, inference_latency_seconds_bucket) > 0.1
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "High inference latency"
      description: "95th percentile inference latency is {{ $value }}s"

  - alert: GPUMemoryHigh
    expr: gpu_memory_usage_percent > 90
    for: 2m
    labels:
      severity: warning
    annotations:
      summary: "High GPU memory usage"
      description: "GPU memory usage is {{ $value }}%"

  - alert: ApplicationDown
    expr: up{job="moe-router"} == 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "MoE Router application is down"
      description: "The MoE Router application has been down for more than 1 minute"
```

### Grafana Dashboard Configuration

```json
{
  "dashboard": {
    "id": null,
    "title": "Self-Evolving MoE Router",
    "tags": ["moe", "evolution", "ai"],
    "timezone": "UTC",
    "refresh": "30s",
    "panels": [
      {
        "title": "Evolution Progress",
        "type": "graph",
        "targets": [
          {
            "expr": "evolution_fitness_best",
            "legendFormat": "Best Fitness"
          },
          {
            "expr": "evolution_fitness_average",
            "legendFormat": "Average Fitness"
          }
        ],
        "yAxes": [
          {
            "label": "Fitness Score",
            "min": 0,
            "max": 1
          }
        ]
      },
      {
        "title": "Expert Utilization",
        "type": "heatmap",
        "targets": [
          {
            "expr": "expert_utilization_ratio",
            "legendFormat": "Expert {{expert_id}}"
          }
        ]
      },
      {
        "title": "Inference Performance",
        "type": "stat",
        "targets": [
          {
            "expr": "histogram_quantile(0.50, inference_latency_seconds_bucket)",
            "legendFormat": "P50 Latency"
          },
          {
            "expr": "histogram_quantile(0.95, inference_latency_seconds_bucket)",
            "legendFormat": "P95 Latency"
          },
          {
            "expr": "inference_throughput_samples",
            "legendFormat": "Throughput"
          }
        ]
      }
    ]
  }
}
```

## Performance Benchmarking

### Benchmark Suite

```python
import time
import torch
import numpy as np
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class BenchmarkResult:
    name: str
    latency_ms: float
    throughput_samples_per_sec: float
    memory_usage_mb: float
    gpu_utilization_percent: float
    accuracy: float

class PerformanceBenchmark:
    """Comprehensive performance benchmarking suite"""
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.results: List[BenchmarkResult] = []
    
    def run_inference_benchmark(self, batch_sizes: List[int], num_iterations: int = 100) -> Dict[int, BenchmarkResult]:
        """Benchmark inference performance across different batch sizes"""
        results = {}
        
        for batch_size in batch_sizes:
            print(f"Benchmarking batch size {batch_size}...")
            
            # Warm up
            self._warmup(batch_size, 10)
            
            # Benchmark
            latencies = []
            throughputs = []
            memory_usage = []
            
            for i in range(num_iterations):
                # Create random input
                input_data = torch.randn(batch_size, 512, 768, device=self.device)
                
                # Measure memory before
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    mem_before = torch.cuda.memory_allocated()
                
                # Time inference
                start_time = time.perf_counter()
                
                with torch.no_grad():
                    output = self.model(input_data)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                end_time = time.perf_counter()
                
                # Calculate metrics
                latency = (end_time - start_time) * 1000  # ms
                throughput = batch_size / (end_time - start_time)  # samples/sec
                
                if torch.cuda.is_available():
                    mem_after = torch.cuda.memory_allocated()
                    memory = (mem_after - mem_before) / (1024 ** 2)  # MB
                else:
                    memory = 0
                
                latencies.append(latency)
                throughputs.append(throughput)
                memory_usage.append(memory)
            
            # Calculate statistics
            result = BenchmarkResult(
                name=f"inference_batch_{batch_size}",
                latency_ms=np.mean(latencies),
                throughput_samples_per_sec=np.mean(throughputs),
                memory_usage_mb=np.mean(memory_usage),
                gpu_utilization_percent=self._get_gpu_utilization(),
                accuracy=0.0  # Would need validation data
            )
            
            results[batch_size] = result
            self.results.append(result)
        
        return results
    
    def run_evolution_benchmark(self, generations: int = 10) -> BenchmarkResult:
        """Benchmark evolution performance"""
        print(f"Benchmarking evolution for {generations} generations...")
        
        start_time = time.perf_counter()
        
        # Run evolution
        for generation in range(generations):
            # Simulate evolution step
            fitness_scores = np.random.random(100)  # Mock fitness evaluation
            time.sleep(0.1)  # Simulate computation time
        
        end_time = time.perf_counter()
        
        total_time = end_time - start_time
        time_per_generation = total_time / generations
        
        result = BenchmarkResult(
            name="evolution_benchmark",
            latency_ms=time_per_generation * 1000,
            throughput_samples_per_sec=generations / total_time,
            memory_usage_mb=self._get_memory_usage(),
            gpu_utilization_percent=self._get_gpu_utilization(),
            accuracy=0.0
        )
        
        self.results.append(result)
        return result
    
    def generate_report(self, output_file: str = "benchmark_report.html"):
        """Generate HTML benchmark report"""
        html_content = """
        <html>
        <head>
            <title>MoE Router Performance Benchmark Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                .metric { font-weight: bold; color: #333; }
            </style>
        </head>
        <body>
            <h1>Self-Evolving MoE Router Benchmark Report</h1>
            <h2>Performance Summary</h2>
            <table>
                <tr>
                    <th>Benchmark</th>
                    <th>Latency (ms)</th>
                    <th>Throughput (samples/sec)</th>
                    <th>Memory Usage (MB)</th>
                    <th>GPU Utilization (%)</th>
                </tr>
        """
        
        for result in self.results:
            html_content += f"""
                <tr>
                    <td>{result.name}</td>
                    <td>{result.latency_ms:.2f}</td>
                    <td>{result.throughput_samples_per_sec:.2f}</td>
                    <td>{result.memory_usage_mb:.2f}</td>
                    <td>{result.gpu_utilization_percent:.1f}</td>
                </tr>
            """
        
        html_content += """
            </table>
        </body>
        </html>
        """
        
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        print(f"Benchmark report saved to {output_file}")
```

## Operational Runbooks

### Evolution Performance Issues

```markdown
## Runbook: Evolution Convergence Problems

### Symptoms
- Evolution fitness not improving over multiple generations
- Low population diversity scores
- Stagnant best fitness metrics

### Diagnosis Steps
1. Check evolution metrics in Grafana dashboard
2. Review evolution logs for errors or warnings
3. Verify population diversity is above 0.1
4. Check mutation and crossover rates

### Resolution Steps
1. Increase mutation rate: `evolution.mutation_rate = 0.2`
2. Restart evolution with fresh population
3. Adjust fitness function if needed
4. Consider increasing population size

### Prevention
- Monitor diversity metrics continuously
- Set up alerts for stagnation
- Regular hyperparameter tuning
```

### Memory and Performance Issues

```markdown
## Runbook: Out of Memory Errors

### Symptoms
- CUDA out of memory errors
- High memory usage alerts
- Application crashes or restarts

### Diagnosis Steps
1. Check GPU memory usage: `nvidia-smi`
2. Review memory metrics in monitoring dashboard
3. Check batch sizes and model parameters
4. Verify memory leaks in application logs

### Resolution Steps
1. Reduce batch size
2. Enable gradient checkpointing
3. Use mixed precision training
4. Restart application to clear memory

### Prevention
- Set memory usage alerts at 80%
- Implement automatic batch size scaling
- Regular memory profiling
```

## Integration with External Systems

### OpenTelemetry Integration

```python
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Configure tracing
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

jaeger_exporter = JaegerExporter(
    agent_host_name="jaeger",
    agent_port=6831,
)

span_processor = BatchSpanProcessor(jaeger_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)

# Usage in application
@tracer.start_as_current_span("evolution_step")
def evolution_step(generation: int):
    with tracer.start_as_current_span("fitness_evaluation"):
        fitness_scores = evaluate_population()
    
    with tracer.start_as_current_span("selection"):
        selected = select_parents(fitness_scores)
    
    with tracer.start_as_current_span("reproduction"):
        new_population = reproduce(selected)
    
    return new_population
```

### Custom Metrics Export

```python
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

class CustomMetricsExporter:
    """Export custom metrics to external systems"""
    
    def __init__(self):
        self.registry = CollectorRegistry()
        self.setup_metrics()
    
    def setup_metrics(self):
        self.evolution_metrics = {
            'generation': Gauge('custom_evolution_generation', 'Current generation', registry=self.registry),
            'fitness': Gauge('custom_evolution_fitness', 'Best fitness', registry=self.registry),
            'diversity': Gauge('custom_evolution_diversity', 'Population diversity', registry=self.registry)
        }
    
    def export_to_pushgateway(self, gateway_url: str, job_name: str):
        """Push metrics to Prometheus Pushgateway"""
        push_to_gateway(gateway_url, job=job_name, registry=self.registry)
    
    def export_to_datadog(self, api_key: str):
        """Export metrics to Datadog"""
        from datadog import initialize, statsd
        
        initialize(api_key=api_key)
        
        # Send metrics
        statsd.gauge('moe.evolution.generation', self.current_generation)
        statsd.gauge('moe.evolution.fitness', self.best_fitness)
        statsd.gauge('moe.evolution.diversity', self.diversity_score)
```

This comprehensive monitoring and observability setup provides full visibility into the Self-Evolving MoE-Router's performance, health, and evolution progress.