#!/bin/bash
# TERRAGON Production Entrypoint Script
# Self-Evolving MoE-Router Container Initialization

set -euo pipefail

# Configuration
TERRAGON_ENV=${TERRAGON_ENV:-production}
TERRAGON_LOG_LEVEL=${TERRAGON_LOG_LEVEL:-INFO}
TERRAGON_WORKERS=${TERRAGON_WORKERS:-4}
TERRAGON_PORT=${TERRAGON_PORT:-8080}

# Logging
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] TERRAGON: $1"
}

# Initialize application
initialize() {
    log "Initializing TERRAGON Self-Evolving MoE-Router"
    log "Environment: ${TERRAGON_ENV}"
    log "Workers: ${TERRAGON_WORKERS}"
    log "Port: ${TERRAGON_PORT}"
    
    # Create necessary directories
    mkdir -p /app/logs /app/data/models /app/data/evolution /app/data/metrics
    
    # Initialize configuration
    if [[ ! -f /app/config/production.json ]]; then
        log "Creating default production configuration"
        python -c "
import json
config = {
    'environment': '${TERRAGON_ENV}',
    'logging': {'level': '${TERRAGON_LOG_LEVEL}', 'file': '/app/logs/terragon.log'},
    'server': {'host': '0.0.0.0', 'port': ${TERRAGON_PORT}, 'workers': ${TERRAGON_WORKERS}},
    'evolution': {
        'population_size': 50,
        'generations': 100,
        'mutation_rate': 0.1,
        'crossover_rate': 0.8,
        'elite_size': 10,
        'parallel_workers': ${TERRAGON_WORKERS},
        'auto_save_interval': 300
    },
    'routing': {
        'cache_size': 10000,
        'cache_ttl': 3600,
        'load_balancing': True,
        'adaptive_routing': True
    },
    'monitoring': {
        'metrics_enabled': True,
        'health_check_interval': 30,
        'performance_logging': True,
        'distributed_tracing': True
    },
    'security': {
        'rate_limiting': True,
        'request_timeout': 30,
        'max_payload_size': 104857600
    }
}
with open('/app/config/production.json', 'w') as f:
    json.dump(config, f, indent=2)
        "
    fi
}

# Pre-flight checks
preflight() {
    log "Running pre-flight checks"
    
    # Check Python environment
    python --version
    
    # Verify core modules can be imported
    python -c "
import sys
sys.path.insert(0, '/app')
try:
    from high_performance_evolution import HighPerformanceEvolutionSystem
    print('✅ High Performance Evolution System: OK')
except ImportError as e:
    print(f'⚠️  High Performance Evolution: {e}')

try:
    from quality_gates_improved import QualityGateExecutor
    print('✅ Quality Gates System: OK')
except ImportError as e:
    print(f'⚠️  Quality Gates: {e}')

try:
    from research_standalone import TERRAGONResearchExecution
    print('✅ Research Execution System: OK')
except ImportError as e:
    print(f'⚠️  Research System: {e}')
    "
    
    # Check disk space
    df -h /app
    
    # Check memory
    free -h
    
    log "Pre-flight checks completed"
}

# Start production server
start_production() {
    log "Starting TERRAGON Production Server"
    
    # Launch production API server
    exec python production_server.py \
        --config /app/config/production.json \
        --workers ${TERRAGON_WORKERS} \
        --port ${TERRAGON_PORT} \
        --env ${TERRAGON_ENV}
}

# Start development server
start_development() {
    log "Starting TERRAGON Development Server"
    
    exec python -m uvicorn development_server:app \
        --host 0.0.0.0 \
        --port ${TERRAGON_PORT} \
        --workers ${TERRAGON_WORKERS} \
        --reload
}

# Run quality gates
run_quality_gates() {
    log "Executing TERRAGON Quality Gates"
    python quality_gates_improved.py
}

# Run research mode
run_research() {
    log "Executing TERRAGON Research Mode"
    python research_standalone.py
}

# Main execution
main() {
    case "${1:-production}" in
        production)
            initialize
            preflight
            start_production
            ;;
        development)
            initialize
            start_development
            ;;
        quality-gates)
            run_quality_gates
            ;;
        research)
            run_research
            ;;
        init)
            initialize
            log "Initialization complete"
            ;;
        preflight)
            preflight
            ;;
        *)
            log "Usage: $0 {production|development|quality-gates|research|init|preflight}"
            exit 1
            ;;
    esac
}

# Handle signals
trap 'log "Received shutdown signal, gracefully stopping..."; exit 0' TERM INT

# Execute main function
main "$@"