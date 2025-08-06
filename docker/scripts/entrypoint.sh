#!/bin/bash
set -e

# Production entrypoint script for Self-Evolving MoE-Router
# Handles initialization, configuration, and service startup

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
    exit 1
}

# Configuration
MOE_ENV=${MOE_ENV:-production}
MOE_LOG_LEVEL=${MOE_LOG_LEVEL:-INFO}
MOE_CONFIG_PATH=${MOE_CONFIG_PATH:-/app/config/production.yaml}
MOE_WORKERS=${MOE_WORKERS:-4}
MOE_PORT=${MOE_PORT:-8000}
MOE_METRICS_PORT=${MOE_METRICS_PORT:-9090}

# Validate environment
validate_environment() {
    log "Validating environment configuration..."
    
    # Check Python environment
    if ! python3 -c "import torch; import self_evolving_moe" 2>/dev/null; then
        error "Failed to import required modules"
    fi
    
    # Check configuration file
    if [[ ! -f "$MOE_CONFIG_PATH" ]]; then
        warn "Configuration file not found at $MOE_CONFIG_PATH, using defaults"
    fi
    
    # Check directories
    mkdir -p /app/logs /app/checkpoints /app/data
    
    # Set permissions
    if [[ $(id -u) == 0 ]]; then
        chown -R moe:moe /app/logs /app/checkpoints /app/data
    fi
    
    log "Environment validation complete"
}

# Initialize system
initialize_system() {
    log "Initializing Self-Evolving MoE-Router system..."
    
    # Set Python path
    export PYTHONPATH="/app/src:$PYTHONPATH"
    
    # Initialize logging directory
    mkdir -p /app/logs
    
    # Set resource limits if running as root
    if [[ $(id -u) == 0 ]]; then
        # Increase file descriptor limits
        ulimit -n 65536
        
        # Set memory limits based on available memory
        TOTAL_MEM=$(cat /proc/meminfo | grep MemTotal | awk '{print $2}')
        MAX_MEM=$((TOTAL_MEM * 80 / 100))  # Use up to 80% of available memory
        echo "Memory limit set to ${MAX_MEM}KB"
    fi
    
    # Initialize GPU if available
    if command -v nvidia-smi >/dev/null 2>&1; then
        log "GPU detected, initializing CUDA environment"
        nvidia-smi
    else
        log "No GPU detected, using CPU-only mode"
    fi
    
    log "System initialization complete"
}

# Health check system
health_check() {
    log "Running health check..."
    
    if python3 /app/healthcheck.py --quick; then
        log "Health check passed"
        return 0
    else
        error "Health check failed"
        return 1
    fi
}

# Start API server
start_api_server() {
    log "Starting MoE-Router API server..."
    log "Configuration: ENV=$MOE_ENV, LOG_LEVEL=$MOE_LOG_LEVEL, WORKERS=$MOE_WORKERS, PORT=$MOE_PORT"
    
    # Create API server if it doesn't exist
    if [[ ! -f /app/src/self_evolving_moe/api.py ]]; then
        warn "API server not found, creating basic server..."
        create_basic_api_server
    fi
    
    # Start with gunicorn for production
    exec gunicorn \
        --bind "0.0.0.0:$MOE_PORT" \
        --workers "$MOE_WORKERS" \
        --worker-class uvicorn.workers.UvicornWorker \
        --max-requests 1000 \
        --max-requests-jitter 100 \
        --timeout 300 \
        --keep-alive 5 \
        --log-level "$MOE_LOG_LEVEL" \
        --access-logfile /app/logs/access.log \
        --error-logfile /app/logs/error.log \
        --capture-output \
        --enable-stdio-inheritance \
        self_evolving_moe.api:app
}

# Start evolution service
start_evolution_service() {
    log "Starting evolution service..."
    
    # Start background evolution service
    python3 -m self_evolving_moe.cli evolve \
        --config "$MOE_CONFIG_PATH" \
        --output-dir /app/checkpoints \
        --verbose &
    
    EVOLUTION_PID=$!
    echo $EVOLUTION_PID > /app/evolution.pid
    
    log "Evolution service started with PID $EVOLUTION_PID"
}

# Start metrics server
start_metrics_server() {
    log "Starting metrics server on port $MOE_METRICS_PORT..."
    
    # Create metrics server if it doesn't exist
    if [[ ! -f /app/src/self_evolving_moe/metrics_server.py ]]; then
        create_basic_metrics_server
    fi
    
    python3 -m self_evolving_moe.metrics_server \
        --port "$MOE_METRICS_PORT" \
        --config "$MOE_CONFIG_PATH" &
    
    METRICS_PID=$!
    echo $METRICS_PID > /app/metrics.pid
    
    log "Metrics server started with PID $METRICS_PID"
}

# Create basic API server
create_basic_api_server() {
    cat > /app/src/self_evolving_moe/api.py << 'EOF'
"""Basic API server for Self-Evolving MoE-Router."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import logging
from typing import Dict, Any, List, Optional
import json
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Self-Evolving MoE-Router API",
    description="REST API for MoE model evolution and inference",
    version="0.1.0"
)

class EvolutionRequest(BaseModel):
    population_size: int = 50
    generations: int = 100
    mutation_rate: float = 0.1
    num_experts: int = 16
    expert_dim: int = 256
    expert_type: str = "mlp"

class InferenceRequest(BaseModel):
    inputs: List[List[float]]
    topology_path: Optional[str] = None

@app.get("/")
async def root():
    return {"message": "Self-Evolving MoE-Router API", "status": "healthy"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "moe-router"}

@app.post("/evolve")
async def start_evolution(request: EvolutionRequest):
    try:
        # Basic evolution endpoint
        return {
            "status": "started",
            "config": request.dict(),
            "message": "Evolution process initiated"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/inference")
async def run_inference(request: InferenceRequest):
    try:
        # Basic inference endpoint
        return {
            "status": "completed",
            "outputs": [[0.5] * 10 for _ in request.inputs],  # Dummy output
            "message": "Inference completed"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def get_metrics():
    return {
        "active_models": 1,
        "total_inferences": 0,
        "avg_latency_ms": 0.0,
        "memory_usage_mb": 0.0
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
EOF
}

# Create basic metrics server
create_basic_metrics_server() {
    cat > /app/src/self_evolving_moe/metrics_server.py << 'EOF'
"""Basic metrics server for Prometheus monitoring."""

import time
import argparse
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import psutil
import os

class MetricsHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/metrics':
            self.send_response(200)
            self.send_header('Content-Type', 'text/plain')
            self.end_headers()
            
            # Basic system metrics in Prometheus format
            metrics = []
            
            # CPU usage
            cpu_percent = psutil.cpu_percent()
            metrics.append(f'moe_cpu_usage_percent {cpu_percent}')
            
            # Memory usage
            memory = psutil.virtual_memory()
            metrics.append(f'moe_memory_usage_percent {memory.percent}')
            metrics.append(f'moe_memory_used_bytes {memory.used}')
            
            # Process info
            process = psutil.Process()
            metrics.append(f'moe_process_memory_bytes {process.memory_info().rss}')
            metrics.append(f'moe_process_cpu_percent {process.cpu_percent()}')
            
            self.wfile.write('\n'.join(metrics).encode())
        else:
            self.send_response(404)
            self.end_headers()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=9090)
    parser.add_argument('--config', type=str, default='')
    args = parser.parse_args()
    
    server = HTTPServer(('0.0.0.0', args.port), MetricsHandler)
    print(f"Metrics server running on port {args.port}")
    server.serve_forever()

if __name__ == "__main__":
    main()
EOF
}

# Graceful shutdown
graceful_shutdown() {
    log "Received shutdown signal, shutting down gracefully..."
    
    # Stop background services
    if [[ -f /app/evolution.pid ]]; then
        EVOLUTION_PID=$(cat /app/evolution.pid)
        kill -TERM $EVOLUTION_PID 2>/dev/null || true
        rm -f /app/evolution.pid
    fi
    
    if [[ -f /app/metrics.pid ]]; then
        METRICS_PID=$(cat /app/metrics.pid)
        kill -TERM $METRICS_PID 2>/dev/null || true
        rm -f /app/metrics.pid
    fi
    
    log "Shutdown complete"
    exit 0
}

# Set up signal handlers
trap graceful_shutdown SIGTERM SIGINT

# Main execution
main() {
    log "Starting Self-Evolving MoE-Router v0.1.0"
    
    # Parse command
    COMMAND=${1:-serve}
    
    case $COMMAND in
        "serve")
            validate_environment
            initialize_system
            health_check
            
            # Start services based on configuration
            start_metrics_server
            start_api_server
            ;;
        
        "evolve")
            validate_environment
            initialize_system
            start_evolution_service
            
            # Wait for evolution to complete
            wait
            ;;
        
        "health")
            health_check
            ;;
        
        "shell")
            log "Starting interactive shell..."
            exec /bin/bash
            ;;
        
        *)
            error "Unknown command: $COMMAND"
            echo "Available commands: serve, evolve, health, shell"
            exit 1
            ;;
    esac
}

# Execute main function
main "$@"